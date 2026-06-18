"""Příprava Experimentu 1: Kvalitativní analýza konceptů — prostatický ViT.

Načte předpočítaná CRP data z FeatureVisualization, identifikuje top-N konceptů
pro cílovou třídu (0=benigní, 1=karcinom) napříč variantami konceptů
a generuje PDF pro hodnotící list (patolog).

Rozvržení PDF
-------------
• Titulní strana  — instrukce a hodnotící postup (pouze hlavní PDF)
• Pro každou vrstvu  — oddělovač vrstvy
  • Pro každou variantu  — oddělovač varianty + jedna strana A4 na šířku pro každý koncept

Heatmapa PDF (--heatmap)
------------------------
Generuje druhé PDF bez instrukcí, s překryvnými teplotními mapami LRP.
Určeno pro výzkumníka nebo experta-patologa, nikoli pro slepé hodnocení.

Předpoklady
-----------
• Spusťte FeatureVisualization.run() pro prostatický dataset před tímto skriptem.
  Dataset MUSÍ být rekonstruován se stejnými parametry jako při FV výpočtu
  (--sample-size, --seed), aby indexy d_c_sorted odpovídaly správným políčkům.
• Cílová třída musí být v RelStats_*/targets.npy.

Příklady použití
----------------
    python tutorials/prepare_exp_1_prostate.py \\
        --fv-path VIT_prostate \\
        --layers model.vit.encoder.layer.9.attention.attention \\
        --class-idx 1 --out exp_1/prostate_cancer.pdf

    python tutorials/prepare_exp_1_prostate.py \\
        --fv-path VIT_prostate \\
        --layers model.vit.encoder.layer.9.attention.attention \\
                  model.vit.encoder.layer.11.attention.attention \\
        --class-idx 1 --sample-size 100000 --heatmap
"""

import argparse
import glob
import os
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image
import openslide   # noqa
import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers.models.vit import modeling_vit
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules

from crp.attribution import AttentionAttribution
from crp.concepts import ChannelConcept, AttentionHeadConcept
from crp.helper import load_maximization, load_statistics, load_stat_targets
from crp.transformer_patches import monkey_patch, monkey_patch_zennit

# ── Konstanty ─────────────────────────────────────────────────────────────────

DATA_PATH         = "/mnt/data/MOU/prostate/tile_level_annotations"
PATCH_SIZE        = 224
SLIDE_LEVEL       = 0
CLASS_NAMES       = {0: "benigní", 1: "karcinom"}
FV_SAMPLE_SIZE    = 100_000   # musí odpovídat hodnotě při FV výpočtu
FV_SEED           = 42        # musí odpovídat hodnotě při FV výpočtu

CONCEPT_MODES = ["head", "token", "dimension"]
MAX_TARGETS   = ["sum", "max"]
VARIANTS      = list(product(CONCEPT_MODES, MAX_TARGETS))

A4_W,  A4_H  = 8.27, 11.69
A4_LW, A4_LH = 11.69, 8.27


# ── Dataset políček (musí odpovídat FV výpočtu) ───────────────────────────────

def _label_from_path(path):
    stem = Path(path).stem
    label = int(stem.split("-")[-1])
    return 1 if label > 0 else 0


class MRXSPatchDataset(torch.utils.data.Dataset):
    """Enumerates all patch positions from all .mrxs slides (identical to create_index script)."""

    def __init__(self, data_path, patch_size=224, level=0, transform=None):
        self.patch_size = patch_size
        self.level      = level
        self.transform  = transform

        slide_paths = sorted(glob.glob(str(Path(data_path) / "*.mrxs")))
        if not slide_paths:
            raise FileNotFoundError(f"No .mrxs files in {data_path}")
        print(f"  Found {len(slide_paths)} slides")

        self.patches = []
        for slide_path in slide_paths:
            label = _label_from_path(slide_path)
            slide = openslide.OpenSlide(slide_path)
            w, h  = slide.level_dimensions[level]
            slide.close()
            for y in range(0, h - patch_size + 1, patch_size):
                for x in range(0, w - patch_size + 1, patch_size):
                    self.patches.append((slide_path, x, y, label))

        labels = [p[3] for p in self.patches]
        print(f"  Total patches: {len(self.patches)} "
              f"(benign={labels.count(0)}, cancer={labels.count(1)})")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        slide_path, x, y, label = self.patches[idx]
        slide = openslide.OpenSlide(slide_path)
        patch = slide.read_region((x, y), self.level, (self.patch_size, self.patch_size))
        slide.close()
        patch = patch.convert("RGB")
        if self.transform:
            patch = self.transform(patch)
        return patch, label


def build_dataset(sample_size, seed, transform):
    """Reconstruct the exact same Subset used during FeatureVisualization.run()."""
    full = MRXSPatchDataset(DATA_PATH, patch_size=PATCH_SIZE, level=SLIDE_LEVEL, transform=transform)
    g = torch.Generator().manual_seed(seed)
    indices = torch.randint(0, len(full), (sample_size,), generator=g).tolist()
    subset  = torch.utils.data.Subset(full, indices)
    print(f"  Subsampled dataset: {len(subset)} patches")
    return subset


def get_image_rgb(dataset, idx):
    img, _ = dataset[int(idx) % len(dataset)]
    return (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")


def get_image_tensor(dataset, idx):
    img, _ = dataset[int(idx) % len(dataset)]
    return img


# ── Model ─────────────────────────────────────────────────────────────────────

class LogitsWrapper(nn.Module):
    """Rozbalí 1 sigmoid logit na 2 logity [-logit, logit] pro třídy 0/1."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logit = self.model(x).logits
        return torch.cat([-logit, logit], dim=-1)


def build_attribution(repo_root, device):
    monkey_patch(modeling_vit, verbose=False)
    monkey_patch_zennit(verbose=False)
    hf = ViTForImageClassification.from_pretrained(
        os.path.join(repo_root, "vit-patch16-224-prostate"),
        attn_implementation="eager",
        local_files_only=True,
    ).to(device).eval()
    hf.requires_grad_(False)
    model      = LogitsWrapper(hf).to(device)
    attribution = AttentionAttribution(model)
    composite   = LayerMapComposite([
        (nn.Conv2d,  z_rules.Gamma(0.25)),
        (nn.Linear,  z_rules.Gamma(0.1)),
    ])
    processor  = ViTImageProcessor.from_pretrained(
        os.path.join(repo_root, "vit-patch16-224-prostate"),
        local_files_only=True,
    )
    normalize  = T.Normalize(mean=processor.image_mean, std=processor.image_std)
    return attribution, composite, normalize


# ── Pomocné funkce pro data ────────────────────────────────────────────────────

def _stats_path(fv_base, concept_mode, max_target, abs_norm=True):
    cm_part  = f"{concept_mode}_" if concept_mode else ""
    norm_str = "normed" if abs_norm else "unnormed"
    return Path(fv_base) / f"RelStats_{cm_part}{max_target}_{norm_str}"


def _relmax_path(fv_base, concept_mode, max_target, abs_norm=True):
    cm_part  = f"{concept_mode}_" if concept_mode else ""
    norm_str = "normed" if abs_norm else "unnormed"
    return Path(fv_base) / f"RelMax_{cm_part}{max_target}_{norm_str}"


def load_variant_data(fv_base, concept_mode, max_target, layer_name, class_idx, abs_norm=True):
    """
    Returns (d_c_relmax, rel_c_stats):
      - d_c_relmax   [SAMPLE_SIZE, num_concepts]  dataset indices from RelMax (all classes)
      - rel_c_stats  [SAMPLE_SIZE, num_concepts]  relevance from RelStats (class-specific)

    Concept ranking uses rel_c_stats; reference images use d_c_relmax so that
    cross-class images are included wherever a concept fires strongly.
    """
    stats_p = _stats_path(fv_base, concept_mode, max_target, abs_norm)
    if not stats_p.exists():
        raise FileNotFoundError(f"Složka RelStats nenalezena: {stats_p}")
    targets = load_stat_targets(stats_p)
    if str(class_idx) not in targets.astype(str):
        raise FileNotFoundError(
            f"Třída {class_idx} není v {stats_p}/targets.npy. "
            "Spusťte nejprve FeatureVisualization.run() pro tuto třídu."
        )
    _, rel_c_sorted, _ = load_statistics(stats_p, layer_name, str(class_idx))

    relmax_p = _relmax_path(fv_base, concept_mode, max_target, abs_norm)
    d_c_sorted, _, _ = load_maximization(relmax_p, layer_name)

    return d_c_sorted, rel_c_sorted


def top_concept_ids(rel_c_sorted, top_n):
    mean_rel = rel_c_sorted.mean(axis=0)
    return np.argsort(mean_rel)[::-1][:top_n].tolist()


# ── Heatmapy ──────────────────────────────────────────────────────────────────

def _mask_map_for(concept_mode):
    if concept_mode == "token":
        return ChannelConcept.mask
    if concept_mode == "dimension":
        return AttentionHeadConcept.dimension_mask
    return None


def compute_heatmaps(attribution, composite, normalize, tensors,
                     concept_id, layer_name, concept_mode, device):
    mask_map = _mask_map_for(concept_mode)
    heatmaps = []
    for t in tensors:
        x = normalize(t).unsqueeze(0).to(device).requires_grad_(True)
        conditions = [{layer_name: [concept_id]}]
        kwargs = dict(start_layer=layer_name, on_device=device, exclude_parallel=False)
        if mask_map is not None:
            kwargs["mask_map"] = mask_map
        attr = attribution(x, conditions, composite, **kwargs)
        heatmaps.append(attr.heatmap[0].cpu().numpy())
    return heatmaps


# ── PDF stránky ───────────────────────────────────────────────────────────────

_INSTRUKCE = """\
EXPERIMENT 1 — Kvalitativní analýza konceptů
Hodnotící list — prostatický ViT

Budete vidět sérii mřížek políček tkáně prostaty.

VÁŠ ÚKOL
─────────
Pro každou mřížku políček rozhodněte, zda políčka sdílejí
rozpoznatelnou společnou histopatologickou vlastnost
(morfologii tkáně, buněčný vzor, architekturu žláz apod.).

Zapište svou odpověď do rámečku na každé stránce:

    ANO  — políčka zřetelně sdílejí společný histologický vzor
    NE   — políčka vypadají nesouvisejícím způsobem nebo vzor je nejasný

Neexistují správné ani špatné odpovědi.
Zakládejte své hodnocení pouze na tom, co vidíte.

─────────────────────────────────────────
Celkem stran hodnocení: {total_pages}
Varianty:  head+sum, head+max
           token+sum, token+max
           dimension+sum, dimension+max
─────────────────────────────────────────
"""


def page_cover(pdf, class_idx, layers, total_concept_pages):
    fig, ax = plt.subplots(figsize=(A4_W, A4_H))
    ax.axis("off")
    fig.patch.set_facecolor("#FFFEF5")

    ax.text(0.5, 0.97, _INSTRUKCE.format(total_pages=total_concept_pages),
            ha="center", va="top", fontsize=11, family="monospace",
            transform=ax.transAxes, linespacing=1.5)

    layer_str = "\n".join(f"  {l}" for l in layers)
    meta = (f"Třída {class_idx}  ·  {CLASS_NAMES[class_idx]}\n"
            f"Vrstvy:\n{layer_str}")
    ax.text(0.5, 0.06, meta, ha="center", va="bottom",
            fontsize=9, color="gray", transform=ax.transAxes)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_layer_divider(pdf, layer_name, page_num, total_pages):
    fig, ax = plt.subplots(figsize=(A4_W, A4_H))
    ax.axis("off")
    fig.patch.set_facecolor("#E9F7EF")

    ax.text(0.5, 0.65, "Vrstva", ha="center", va="center",
            fontsize=16, color="#555", transform=ax.transAxes)
    ax.text(0.5, 0.52, layer_name, ha="center", va="center",
            fontsize=14, fontweight="bold", color="#1E8449",
            transform=ax.transAxes)
    ax.text(0.5, 0.05, f"Strana {page_num} z {total_pages}",
            ha="center", va="bottom", fontsize=9, color="#888",
            transform=ax.transAxes)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_variant_divider(pdf, concept_mode, max_target, concept_ids,
                         layer_name, page_num, total_pages):
    fig, ax = plt.subplots(figsize=(A4_W, A4_H))
    ax.axis("off")
    fig.patch.set_facecolor("#E8F4FD")

    ax.text(0.5, 0.72, "Varianta", ha="center", va="center",
            fontsize=16, color="#555", transform=ax.transAxes)
    ax.text(0.5, 0.60, f"{concept_mode}  +  {max_target}",
            ha="center", va="center", fontsize=30, fontweight="bold",
            color="#1A5276", transform=ax.transAxes)
    ids_str = "   ".join(f"#{c}" for c in concept_ids)
    ax.text(0.5, 0.46, f"ID nejlepších konceptů: {ids_str}",
            ha="center", va="center", fontsize=13, color="#333",
            transform=ax.transAxes)
    ax.text(0.5, 0.35, f"Vrstva: {layer_name}",
            ha="center", va="center", fontsize=10, color="#666",
            transform=ax.transAxes)
    ax.text(0.5, 0.05, f"Strana {page_num} z {total_pages}",
            ha="center", va="bottom", fontsize=9, color="#888",
            transform=ax.transAxes)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _build_grid_axes(fig, n_rows, n_cols, grid_top, grid_bot):
    axes = []
    for r in range(n_rows):
        for c in range(n_cols):
            left   = 0.01 + c * (0.98 / n_cols)
            width  = 0.97 / n_cols - 0.01
            bottom = grid_bot + (n_rows - 1 - r) * ((grid_top - grid_bot) / n_rows)
            height = (grid_top - grid_bot) / n_rows - 0.01
            ax = fig.add_axes([left, bottom, width, height])
            axes.append(ax)
    return axes


def _draw_images(axes, images, heatmaps=None):
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
            if heatmaps is not None and i < len(heatmaps):
                h = heatmaps[i]
                vmax = max(float(np.abs(h).max()), 1e-10)
                ax.imshow(h, cmap="RdBu_r", alpha=0.45,
                          vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5); spine.set_edgecolor("#CCC")
        else:
            ax.set_visible(False)


def save_tiles_png(images, concept_id, concept_mode, max_target, layer_name, image_dir, heatmaps=None):
    """Save individual tile images as PNG files using PIL (no matplotlib overhead)."""
    safe_layer = layer_name.replace(".", "_")
    suffix = "_heatmap" if heatmaps is not None else ""
    out_dir = Path(image_dir) / safe_layer / f"{concept_mode}_{max_target}" / f"concept_{concept_id}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        pil_img = Image.fromarray(img)
        if heatmaps is not None and i < len(heatmaps):
            h = heatmaps[i]
            vmax = max(float(np.abs(h).max()), 1e-10)
            h_norm = ((h / vmax + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            overlay = Image.fromarray(np.stack([h_norm, np.zeros_like(h_norm), 255 - h_norm], axis=-1))
            overlay = overlay.resize(pil_img.size)
            pil_img = Image.blend(pil_img.convert("RGBA"), overlay.convert("RGBA"), alpha=0.45).convert("RGB")
        pil_img.save(out_dir / f"tile_{i:02d}.png")


def page_concept_grid(pdf, images, concept_id, concept_rank, top_n,
                      concept_mode, max_target, layer_name,
                      page_num, total_pages, n_cols=3, heatmaps=None,
                      image_dir=None):
    n_rows   = (len(images) + n_cols - 1) // n_cols
    rating_h = 0.00 if heatmaps is not None else 0.11
    grid_top = 0.88
    grid_bot = rating_h + 0.01

    fig = plt.figure(figsize=(A4_LW, A4_LH))
    fig.patch.set_facecolor("white")

    axes = _build_grid_axes(fig, n_rows, n_cols, grid_top, grid_bot)
    _draw_images(axes, images, heatmaps)

    fig.text(0.5, 0.955,
             f"Varianta: {concept_mode}+{max_target}   |   "
             f"Koncept {concept_rank} z {top_n}  (ID #{concept_id})   |   "
             f"Vrstva: {layer_name}",
             ha="center", va="top", fontsize=11, fontweight="bold")

    if heatmaps is not None:
        fig.text(0.5, 0.922, "Podmíněné atribuční heatmapy (LRP)",
                 ha="center", va="top", fontsize=9, color="#555")
    else:
        fig.text(0.5, 0.922,
                 "Sdílejí tato políčka rozpoznatelnou histopatologickou společnou vlastnost?",
                 ha="center", va="top", fontsize=10, color="#333")

        box_ax = fig.add_axes([0.15, 0.01, 0.70, rating_h - 0.01])
        box_ax.axis("off")
        box_ax.set_xlim(0, 1); box_ax.set_ylim(0, 1)
        for xi, label in [(0.25, "ANO"), (0.75, "NE")]:
            box_ax.add_patch(mpatches.FancyBboxPatch(
                (xi - 0.12, 0.10), 0.24, 0.80,
                boxstyle="round,pad=0.02",
                linewidth=1.5, edgecolor="#444", facecolor="white"))
            box_ax.text(xi, 0.50, label,
                        ha="center", va="center",
                        fontsize=16, fontweight="bold", color="#222")
        fig.add_artist(plt.Line2D(
            [0.05, 0.95], [rating_h, rating_h],
            transform=fig.transFigure,
            color="#CCC", linewidth=0.8, zorder=5))

    fig.text(0.97, 0.01, f"{page_num}/{total_pages}",
             ha="right", va="bottom", fontsize=8, color="#AAA")

    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)

    if image_dir is not None:
        save_tiles_png(images, concept_id, concept_mode, max_target, layer_name, image_dir, heatmaps)


# ── Hlavní funkce ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generuje PDF Experimentu 1 pro prostatický ViT"
    )
    parser.add_argument("--fv-path", required=True,
                        help="Základní adresář předpočítaných dat FeatureVisualization")
    parser.add_argument("--layers", nargs="+", required=True, metavar="LAYER",
                        help="Jedna nebo více vrstev použitých při FeatureVisualization.run()")
    parser.add_argument("--class-idx", type=int, required=True, choices=[0, 1],
                        help="Cílová třída: 0=benigní, 1=karcinom")
    parser.add_argument("--top-n", type=int, default=3,
                        help="Počet nejlepších konceptů na variantu (výchozí: 3)")
    parser.add_argument("--n-refs", type=int, default=9,
                        help="Referenčních políček na koncept (výchozí: 9)")
    parser.add_argument("--sample-size", type=int, default=FV_SAMPLE_SIZE,
                        help=f"Subset size used during FeatureVisualization.run() (default: {FV_SAMPLE_SIZE})")
    parser.add_argument("--seed", type=int, default=FV_SEED,
                        help=f"Random seed used during FeatureVisualization.run() (default: {FV_SEED})")
    parser.add_argument("--no-abs-norm", action="store_true",
                        help="Načíst nenormovaná data RelStats (výchozí: normovaná)")
    parser.add_argument("--heatmap", action="store_true",
                        help="Generovat také PDF s teplotními mapami (pro výzkumníka)")
    parser.add_argument("--out", default=None,
                        help="Výstupní cesta PDF (výchozí: exp_1_prostate_<třída>.pdf)")
    parser.add_argument("--image-dir", default=None, metavar="DIR",
                        help="If set, save each concept grid and individual tiles as PNG files in DIR")
    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    repo_root = os.path.abspath(os.getcwd())
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    abs_norm = not args.no_abs_norm
    n_cols   = 3 if args.n_refs == 9 else 4

    out_main = args.out or f"exp_1_prostate_{CLASS_NAMES[args.class_idx]}.pdf"
    out_heat = out_main.replace(".pdf", "_heatmap.pdf")
    Path(out_main).parent.mkdir(parents=True, exist_ok=True)

    print("Načítám dataset políček ...")
    transform = T.Compose([T.Resize((PATCH_SIZE, PATCH_SIZE)), T.ToTensor()])
    dataset   = build_dataset(args.sample_size, args.seed, transform)

    attribution, composite, normalize = None, None, None
    if args.heatmap:
        print("Načítám model pro heatmapy ...")
        attribution, composite, normalize = build_attribution(repo_root, device)

    # ── Předvýpočet dat pro všechny vrstvy a varianty ────────────────────────
    plan = {}
    for layer_name in args.layers:
        entries = []
        for concept_mode, max_target in VARIANTS:
            try:
                d_c, rel_c = load_variant_data(
                    args.fv_path, concept_mode, max_target,
                    layer_name, args.class_idx, abs_norm)
            except FileNotFoundError as e:
                print(f"  Přeskakuji {layer_name} / {concept_mode}+{max_target}: {e}")
                continue
            c_ids = top_concept_ids(rel_c, args.top_n)
            entries.append((concept_mode, max_target, c_ids, d_c))
            print(f"  {layer_name} / {concept_mode}+{max_target}: top = {c_ids}")
        if entries:
            plan[layer_name] = entries

    if not plan:
        raise SystemExit("Nenalezena žádná data. Spusťte nejprve FeatureVisualization.run().")

    n_layer_blocks  = sum(1 + len(e) * (1 + args.top_n) for e in plan.values())
    total_pdf_pages = 1 + n_layer_blocks
    total_heat_pages = n_layer_blocks
    total_concept_pages = sum(len(e) * args.top_n for e in plan.values())

    print(f"\nGeneruji hlavní PDF: {out_main}")
    pdfs = [PdfPages(out_main)]
    if args.heatmap:
        print(f"Generuji heatmapa PDF: {out_heat}")
        pdfs.append(PdfPages(out_heat))

    try:
        page_main = 1
        page_heat = 1

        page_cover(pdfs[0], args.class_idx, args.layers, total_concept_pages)
        page_main += 1

        for layer_name, entries in plan.items():
            print(f"\n  Vrstva: {layer_name}")

            page_layer_divider(pdfs[0], layer_name, page_main, total_pdf_pages)
            page_main += 1
            if args.heatmap:
                page_layer_divider(pdfs[1], layer_name, page_heat, total_heat_pages)
                page_heat += 1

            for concept_mode, max_target, c_ids, d_c_sorted in entries:
                print(f"    Varianta {concept_mode}+{max_target} ...")

                page_variant_divider(pdfs[0], concept_mode, max_target, c_ids,
                                     layer_name, page_main, total_pdf_pages)
                page_main += 1
                if args.heatmap:
                    page_variant_divider(pdfs[1], concept_mode, max_target, c_ids,
                                         layer_name, page_heat, total_heat_pages)
                    page_heat += 1

                for rank, c_id in enumerate(c_ids, start=1):
                    print(f"      Koncept {rank}/{args.top_n} (ID #{c_id}) ...")
                    indices = d_c_sorted[:args.n_refs, c_id]
                    images  = [get_image_rgb(dataset, i) for i in indices]
                    print(f"        DEBUG loaded {len(images)} tiles, shapes={[img.shape for img in images]}, mean={[img.mean().round(1) for img in images]}")
                    tensors = [get_image_tensor(dataset, i) for i in indices]

                    page_concept_grid(pdfs[0], images, c_id, rank, args.top_n,
                                      concept_mode, max_target, layer_name,
                                      page_main, total_pdf_pages, n_cols,
                                      image_dir=args.image_dir)
                    page_main += 1

                    if args.heatmap:
                        hmaps = compute_heatmaps(attribution, composite, normalize,
                                                 tensors, c_id, layer_name,
                                                 concept_mode, device)
                        page_concept_grid(pdfs[1], images, c_id, rank, args.top_n,
                                          concept_mode, max_target, layer_name,
                                          page_heat, total_heat_pages, n_cols,
                                          heatmaps=hmaps, image_dir=args.image_dir)
                        page_heat += 1
    finally:
        for p in pdfs:
            p.close()

    print(f"\nHotovo. Hlavní PDF: {out_main}  ({total_pdf_pages} stran)")
    if args.heatmap:
        print(f"         Heatmapa PDF: {out_heat}  ({total_heat_pages} stran)")


if __name__ == "__main__":
    main()
