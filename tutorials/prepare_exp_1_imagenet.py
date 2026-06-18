"""Příprava Experimentu 1: Kvalitativní analýza konceptů — referenční obrázky pro lidské posuzování.

Načte předpočítaná CRP data z FeatureVisualization, identifikuje top-N konceptů
pro cílovou třídu napříč šesti variantami (head/token/dimension × sum/max)
a generuje PDF pro hodnotící list lidského posuzování.

Možnosti
--------
  Hlavní PDF (--out):  instrukce + mřížky obrázků + rámečky ANO/NE (pro respondenty)
  Heatmapa PDF (--heatmap):  bez instrukcí, s překryvnými teplotními mapami (pro výzkumníka)

Rozvržení PDF
-------------
• Titulní strana  — instrukce a hodnotící postup (pouze hlavní PDF)
• Pro každou vrstvu  — oddělovač vrstvy
  • Pro každou variantu  — oddělovač varianty + jedna strana A4 na šířku pro každý koncept

Předpoklady
-----------
Před spuštěním spusťte FeatureVisualization.run() pro cílovou třídu.
Cílová třída musí být v RelStats_*/targets.npy.

Příklady použití
----------------
    python tutorials/prepare_exp_1_imagenet.py \\
        --fv-path VIT_B16_ImageNet \\
        --layers encoder.layers.encoder_layer_9.self_attention \\
                  encoder.layers.encoder_layer_11.self_attention \\
        --class-idx 0 --out exp_1/eval_class0.pdf

    python tutorials/prepare_exp_1_imagenet.py \\
        --fv-path VIT_B16_ImageNet \\
        --layers encoder.layers.encoder_layer_9.self_attention \\
        --class-idx 0 --heatmap
"""

import argparse
import os
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from crp.visualization import vis_img_heatmap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.models import vision_transformer
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules

from crp.attribution import AttentionAttribution
from crp.concepts import ChannelConcept, AttentionHeadConcept
from crp.helper import load_maximization, load_statistics, load_stat_targets
from crp.transformer_patches import monkey_patch, monkey_patch_zennit
from crp.visualization import FeatureVisualization

# ── Konstanty ─────────────────────────────────────────────────────────────────

DATA_PATH = "ImageNet_data"

CONCEPT_MODES = ["head", "token", "dimension"]
MAX_TARGETS   = ["sum", "max"]
VARIANTS      = list(product(CONCEPT_MODES, MAX_TARGETS))

A4_W,  A4_H  = 8.27, 11.69   # na výšku (palce)
A4_LW, A4_LH = 11.69, 8.27   # na šířku (palce)

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


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


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_dataset():
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    return torchvision.datasets.ImageNet(DATA_PATH, split="val", transform=transform)


def get_image_rgb(dataset, idx):
    img, _ = dataset[int(idx)]
    return (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")


def get_image_tensor(dataset, idx):
    img, _ = dataset[int(idx)]
    return img   # [C, H, W] in [0, 1]


# ── Model a atribuce (pro heatmapy) ──────────────────────────────────────────

def build_attribution(device):
    monkey_patch(vision_transformer, verbose=False)
    monkey_patch_zennit(verbose=False)
    weights = vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
    model   = vision_transformer.vit_b_16(weights=weights).eval().to(device)
    model.requires_grad_(False)
    return AttentionAttribution(model), LayerMapComposite([
        (nn.Conv2d,  z_rules.Gamma(0.25)),
        (nn.Linear,  z_rules.Gamma(0.1)),
    ])


def build_feature_visualization(attribution, dataset, fv_path, layer_names):
    """Initialize FeatureVisualization for heatmap computation."""
    def preprocess_fn(img_tensor):
        return NORMALIZE(img_tensor)

    ahc = AttentionHeadConcept()
    layer_map = {layer_name: ahc for layer_name in layer_names}
    return FeatureVisualization(
        attribution, dataset, layer_map,
        preprocess_fn=preprocess_fn,
        path=fv_path
    )


def _mask_map_for(concept_mode):
    if concept_mode == "token":
        return ChannelConcept.mask
    if concept_mode == "dimension":
        return AttentionHeadConcept.dimension_mask
    return None   # AttentionAttribution default = AttentionHeadConcept.mask


def get_heatmaps_from_fv(fv, concept_ids, layer_name, concept_mode, max_target,
                         composite, r_range):
    """Get heatmap images using FeatureVisualization.get_max_reference."""
    try:
        ref_images = fv.get_max_reference(
            concept_ids, layer_name, mode="relevance", r_range=r_range,
            composite=composite, concept_mode=concept_mode, max_target=max_target,
            plot_fn=vis_img_heatmap
        )
        return ref_images
    except Exception as e:
        print(f"Warning: Failed to get heatmaps from FV: {e}")
        return {}


# ── PDF stránky ───────────────────────────────────────────────────────────────

_INSTRUKCE = """\
EXPERIMENT 1 — Kvalitativní analýza konceptů
Hodnotící list

Budete vidět sérii mřížek obrázků.

VÁŠ ÚKOL
─────────
Pro každou mřížku obrázků rozhodněte, zda obrázky sdílejí
rozpoznatelnou vizuální společnou vlastnost (texturu, část objektu,
barevný vzor, typ scény apod.).

Zapište svou odpověď do rámečku na každé stránce:

    ANO  — obrázky zřetelně sdílejí společný vizuální vzor
    NE   — obrázky vypadají nesouvisejícím způsobem nebo vzor je nejasný

Neexistují správné ani špatné odpovědi.
Zakládejte své hodnocení pouze na tom, co vidíte.

─────────────────────────────────────────
Celkem stran hodnocení: {total_pages}
Varianty:  head+sum, head+max
           token+sum, token+max
           dimension+sum, dimension+max
─────────────────────────────────────────
"""


def page_cover(pdf, class_name, class_idx, layers, total_concept_pages):
    fig, ax = plt.subplots(figsize=(A4_W, A4_H))
    ax.axis("off")
    fig.patch.set_facecolor("#FFFEF5")

    ax.text(0.5, 0.97, _INSTRUKCE.format(total_pages=total_concept_pages),
            ha="center", va="top", fontsize=11, family="monospace",
            transform=ax.transAxes, linespacing=1.5)

    layer_str = "\n".join(f"  {l}" for l in layers)
    meta = f"Třída {class_idx}  ·  {class_name}\nVrstvy:\n{layer_str}"
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
            fontsize=18, fontweight="bold", color="#1E8449",
            transform=ax.transAxes, wrap=True)
    ax.text(0.5, 0.05, f"Strana {page_num} z {total_pages}",
            ha="center", va="bottom", fontsize=9, color="#888",
            transform=ax.transAxes)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_variant_divider(pdf, concept_mode, max_target, concept_ids, layer_name,
                         page_num, total_pages):
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
                if hasattr(h, "mode"):
                    ax.imshow(h, alpha=0.45, interpolation="nearest")
                else:
                    h_arr = np.asarray(h)
                    if h_arr.ndim == 3 and h_arr.shape[-1] in (3, 4):
                        ax.imshow(h_arr, alpha=0.45, interpolation="nearest")
                    else:
                        vmax = max(float(np.abs(h_arr).max()), 1e-10)
                        ax.imshow(h_arr, cmap="RdBu_r", alpha=0.45,
                                  vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5); spine.set_edgecolor("#CCC")
        else:
            ax.set_visible(False)

def page_concept_grid(pdf, images, concept_id, concept_rank, top_n,
                      concept_mode, max_target, layer_name,
                      page_num, total_pages, n_cols=3,
                      heatmaps=None, side_by_side=False):
    if heatmaps is not None and side_by_side:
        paired = []
        for img, hm in zip(images, heatmaps):
            paired.extend([img, hm])
        images = paired
        heatmaps = None
        n_cols = n_cols * 2

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
                 "Sdílejí tyto obrázky rozpoznatelnou vizuální společnou vlastnost?",
                 ha="center", va="top", fontsize=10, color="#333")

        # Rating box
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

    pdf.savefig(fig)
    plt.close(fig)


# ── Hlavní funkce ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generuje PDF Experimentu 1 s referenčními obrázky pro lidské posuzování"
    )
    parser.add_argument("--fv-path", required=True,
                        help="Základní adresář předpočítaných dat FeatureVisualization")
    parser.add_argument("--layers", nargs="+", required=True, metavar="LAYER",
                        help="Jedna nebo více vrstev použitých při FeatureVisualization.run()")
    parser.add_argument("--class-idx", type=int, required=True,
                        help="Index cílové třídy ImageNet (musí být v RelStats/targets.npy)")
    parser.add_argument("--top-n", type=int, default=3,
                        help="Počet nejlepších konceptů na variantu (výchozí: 3)")
    parser.add_argument("--n-refs", type=int, default=9,
                        help="Referenčních obrázků na koncept — 9 → mřížka 3×3, 8 → mřížka 2×4 (výchozí: 9)")
    parser.add_argument("--no-abs-norm", action="store_true",
                        help="Načíst nenormovaná data RelStats (výchozí: normovaná)")
    parser.add_argument("--heatmap", action="store_true",
                        help="Generovat také PDF s teplotními mapami (pro výzkumníka)")
    parser.add_argument("--out", default=None,
                        help="Výstupní cesta PDF (výchozí: exp_1_<třída>.pdf)")
    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    abs_norm = not args.no_abs_norm
    if args.n_refs == 9:
        n_cols = 3
    elif args.n_refs == 8:
        n_cols = 4
    elif args.n_refs == 4:
        n_cols = 2
    else:
        n_cols = max(1, min(4, args.n_refs))
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    out_main = args.out or f"exp_1_{args.class_idx}.pdf"
    out_heat = out_main.replace(".pdf", "_heatmap.pdf")
    Path(out_main).parent.mkdir(parents=True, exist_ok=True)

    print("Načítám dataset ...")
    dataset    = build_dataset()
    class_name = dataset.classes[args.class_idx][0]
    print(f"Třída {args.class_idx}: {class_name}")

    attribution, composite = None, None
    fv = None
    if args.heatmap:
        print("Načítám model a FeatureVisualization ...")
        attribution, composite = build_attribution(device)
        fv = build_feature_visualization(attribution, dataset, args.fv_path, args.layers)

    # ── Předvýpočet dat pro všechny vrstvy a varianty ────────────────────────
    # plan[layer] = [(concept_mode, max_target, c_ids, d_c_sorted), ...]
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

    # Page count:
    # main PDF: cover + per layer: layer_divider + per variant: variant_divider + top_n concept pages
    n_layer_blocks = sum(
        1 + len(entries) * (1 + args.top_n)
        for entries in plan.values()
    )
    total_pdf_pages  = 1 + n_layer_blocks
    total_heat_pages = n_layer_blocks   # no cover in heatmap PDF

    total_concept_pages = sum(len(e) * args.top_n for e in plan.values())

    print(f"\nGeneruji hlavní PDF: {out_main}")
    pdfs = [PdfPages(out_main)]
    if args.heatmap:
        print(f"Generuji heatmapa PDF: {out_heat}")
        pdfs.append(PdfPages(out_heat))

    try:
        page_main = 1
        page_heat = 1

        # Cover (main only)
        page_cover(pdfs[0], class_name, args.class_idx, args.layers, total_concept_pages)
        page_main += 1

        for layer_name, entries in plan.items():
            print(f"\n  Vrstva: {layer_name}")

            # Layer divider
            page_layer_divider(pdfs[0], layer_name, page_main, total_pdf_pages)
            page_main += 1
            if args.heatmap:
                page_layer_divider(pdfs[1], layer_name, page_heat, total_heat_pages)
                page_heat += 1

            for concept_mode, max_target, c_ids, d_c_sorted in entries:
                print(f"    Varianta {concept_mode}+{max_target} ...")

                # Variant divider
                page_variant_divider(pdfs[0], concept_mode, max_target, c_ids,
                                     layer_name, page_main, total_pdf_pages)
                page_main += 1
                if args.heatmap:
                    page_variant_divider(pdfs[1], concept_mode, max_target, c_ids,
                                         layer_name, page_heat, total_heat_pages)
                    page_heat += 1

                for rank, c_id in enumerate(c_ids, start=1):
                    print(f"      Koncept {rank}/{args.top_n} (ID #{c_id}) ...")
                    indices  = d_c_sorted[:args.n_refs, c_id]
                    images   = [get_image_rgb(dataset, i) for i in indices]
                    tensors  = [get_image_tensor(dataset, i) for i in indices]

                    page_concept_grid(pdfs[0], images, c_id, rank, args.top_n,
                                      concept_mode, max_target, layer_name,
                                      page_main, total_pdf_pages, n_cols)
                    page_main += 1

                    if args.heatmap and fv is not None:
                        ref_dict = get_heatmaps_from_fv(
                            fv, [c_id], layer_name, concept_mode, max_target,
                            composite, r_range=(0, args.n_refs)
                        )
                        ref_val = ref_dict.get(c_id)
                        if isinstance(ref_val, tuple) and len(ref_val) == 2:
                            heat_images, heat_overlays = ref_val
                            page_concept_grid(pdfs[1], heat_images, c_id, rank, args.top_n,
                                              concept_mode, max_target, layer_name,
                                              page_heat, total_heat_pages, n_cols,
                                              heatmaps=heat_overlays, side_by_side=True)
                        elif ref_val is not None:
                            page_concept_grid(pdfs[1], ref_val, c_id, rank, args.top_n,
                                              concept_mode, max_target, layer_name,
                                              page_heat, total_heat_pages, n_cols)
                        else:
                            page_concept_grid(pdfs[1], images, c_id, rank, args.top_n,
                                              concept_mode, max_target, layer_name,
                                              page_heat, total_heat_pages, n_cols)
                        page_heat += 1
    finally:
        for p in pdfs:
            p.close()

    print(f"\nHotovo. Hlavní PDF: {out_main}  ({total_pdf_pages} stran)")
    if args.heatmap:
        print(f"         Heatmapa PDF: {out_heat}  ({total_heat_pages} stran)")


if __name__ == "__main__":
    main()
