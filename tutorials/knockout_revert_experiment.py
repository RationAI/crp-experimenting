"""Experiment 2 — REVERT variant.

Compares the CRP top-down knockout curve (most-relevant first) against the
CRP bottom-up knockout curve (least-relevant first, i.e. reversed CRP ranking).

If CRP ranks concepts faithfully:
    knocking the MOST relevant first  → steep early drop
    knocking the LEAST relevant first → shallow early drop, sharp late drop
The two curves must meet at k=0 (no knockout, drop=0) and k=H (all knocked out,
same model state, same drop).  The signed AUC difference
    AUC_signed = ∫ (mean_TOP − mean_REV) dk
is therefore a direct test of CRP ranking quality, independent of any random
baseline.  Larger AUC_signed = stronger evidence that the ranking is meaningful.

Only the signed AUC is reported (trapezoidal, with k=0 anchor).

H0 / interpretation:
    AUC_signed = 0  ⇔  CRP ranking carries no monotone information about
    knockout-induced probability drop.  Sign and magnitude indicate direction
    and strength of the effect.

CLI:
    python knockout_revert_experiment.py --concept-mode token --max-target max
    python knockout_revert_experiment.py --class-idx 60 --concept-mode head --max-target sum
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.models import vision_transformer
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules

from crp.attribution import AttentionAttribution
from crp.transformer_patches import monkey_patch, monkey_patch_zennit


# ── Architecture constants for ViT-B/16 ───────────────────────────────────────
IMAGES_PER_CLASS = 50
DATA_PATH = "ImageNet_data"
NUM_HEADS = 12
HIDDEN_DIM = 768
HEAD_DIM = HIDDEN_DIM // NUM_HEADS
NUM_TOKENS = 197
DEFAULT_LAYER = "encoder.layers.encoder_layer_9.self_attention"

CONCEPT_MODES = ["head", "token", "dimension"]
MAX_TARGETS = ["sum", "max"]

_NUM_CONCEPTS = {"head": NUM_HEADS, "token": NUM_TOKENS, "dimension": HIDDEN_DIM}


# ── Model / data setup ────────────────────────────────────────────────────────

def build_model(device):
    weights = vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
    model = vision_transformer.vit_b_16(weights=weights).eval().to(device)
    model.requires_grad_(False)
    return model


def build_composite():
    return LayerMapComposite([
        (nn.Conv2d, z_rules.Gamma(0.25)),
        (nn.Linear, z_rules.Gamma(0.1)),
    ])


def build_dataset():
    """ImageFolder over <DATA_PATH>/val. Subdirs sorted alphabetically (wnid order
    matches canonical ImageNet class index), so dataset[class_idx*50 + i] yields
    the i-th val image of class class_idx — same indexing as torchvision.ImageNet."""
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    val_dir = os.path.join(DATA_PATH, "val")
    return torchvision.datasets.ImageFolder(val_dir, transform=transform)


# ── Relevance aggregation ─────────────────────────────────────────────────────

def aggregate_relevance(relevance, concept_mode, max_target):
    """relevance: (1, seq_len, hidden_dim) → (H,) normalized per-concept scores."""
    rel = relevance[0]

    if concept_mode == "head":
        rel_h = rel.view(-1, NUM_HEADS, HEAD_DIM).permute(1, 0, 2).reshape(NUM_HEADS, -1)
        scores = rel_h.sum(dim=-1) if max_target == "sum" else rel_h.max(dim=-1).values
    elif concept_mode == "token":
        scores = rel.sum(dim=-1) if max_target == "sum" else rel.max(dim=-1).values
    elif concept_mode == "dimension":
        scores = rel.sum(dim=0) if max_target == "sum" else rel.max(dim=0).values
    else:
        raise ValueError(f"Unknown concept_mode '{concept_mode}'")

    norm = scores.abs().sum().clamp(min=1e-10)
    return scores / norm


# ── Knockout hooks ────────────────────────────────────────────────────────────

def head_knockout_hook(ids):
    def hook(module, input, output):
        out = output[0].clone() if isinstance(output, tuple) else output.clone()
        for h in ids:
            out[:, :, h * HEAD_DIM:(h + 1) * HEAD_DIM] = 0.0
        return (out,) + output[1:] if isinstance(output, tuple) else out
    return hook


def token_knockout_hook(ids):
    def hook(module, input, output):
        out = output[0].clone() if isinstance(output, tuple) else output.clone()
        for t in ids:
            out[:, t, :] = 0.0
        return (out,) + output[1:] if isinstance(output, tuple) else out
    return hook


def dimension_knockout_hook(ids):
    def hook(module, input, output):
        out = output[0].clone() if isinstance(output, tuple) else output.clone()
        for d in ids:
            out[:, :, d] = 0.0
        return (out,) + output[1:] if isinstance(output, tuple) else out
    return hook


_MAKE_HOOK = {
    "head": head_knockout_hook,
    "token": token_knockout_hook,
    "dimension": dimension_knockout_hook,
}


def forward_prob(model, data, class_idx, layer_module, hook_fn):
    handle = layer_module.register_forward_hook(hook_fn)
    with torch.no_grad():
        prob = torch.softmax(model(data), dim=-1)[0, class_idx].item()
    handle.remove()
    return prob


# ── Experiment ────────────────────────────────────────────────────────────────

def run(class_idx, layer_name, concept_mode, max_target, device):
    print(f"Building model on {device} ...")
    model = build_model(device)
    composite = build_composite()
    preprocessing = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = build_dataset()

    raw_class = dataset.classes[class_idx]
    class_name = raw_class[0] if isinstance(raw_class, (list, tuple)) else raw_class
    print(f"Class {class_idx}: {class_name}")

    layer_module = dict(model.named_modules())[layer_name]
    attribution = AttentionAttribution(model)

    H = _NUM_CONCEPTS[concept_mode]
    make_hook = _MAKE_HOOK[concept_mode]
    M = IMAGES_PER_CLASS
    start_idx = class_idx * IMAGES_PER_CLASS

    top_drops = np.zeros((M, H))
    rev_drops = np.zeros((M, H))
    baseline_probs = np.zeros(M)

    print(f"Running REVERT knockout: {M} images, {H} concepts, "
          f"mode={concept_mode}+{max_target}, layer={layer_name}")

    for img_i in range(M):
        if img_i % 5 == 0:
            print(f"  image {img_i + 1}/{M} ...")

        raw_img, _ = dataset[start_idx + img_i]
        data = preprocessing(raw_img).unsqueeze(0).to(device)

        data_g = data.clone().requires_grad_(True)
        attr = attribution(
            data_g,
            conditions=[{"y": [class_idx]}],
            composite=composite,
            record_layer=[layer_name],
            mask_map=None,
        )
        p_base = torch.softmax(attr.prediction, dim=-1)[0, class_idx].item()
        baseline_probs[img_i] = p_base

        relevance = attr.relevances[layer_name].detach().cpu()
        scores = aggregate_relevance(relevance, concept_mode, max_target)
        ranked_top = torch.argsort(scores, descending=True).tolist()  # most relevant first
        ranked_rev = list(reversed(ranked_top))                       # least relevant first

        for k in range(1, H + 1):
            p_top = forward_prob(model, data, class_idx, layer_module, make_hook(ranked_top[:k]))
            top_drops[img_i, k - 1] = p_base - p_top

            p_rev = forward_prob(model, data, class_idx, layer_module, make_hook(ranked_rev[:k]))
            rev_drops[img_i, k - 1] = p_base - p_rev

    return dict(
        top_drops=top_drops,
        rev_drops=rev_drops,
        baseline_probs=baseline_probs,
        class_idx=class_idx,
        class_name=class_name,
        layer_name=layer_name,
        concept_mode=concept_mode,
        max_target=max_target,
        H=H,
        M=M,
    )


# ── Metric + plot ─────────────────────────────────────────────────────────────

def compute_signed_auc(top_drops, rev_drops):
    """Trapezoidal signed AUC with k=0 anchor (drop=0 for both curves)."""
    M, H = top_drops.shape
    zeros = np.zeros((M, 1))
    top_full = np.concatenate([zeros, top_drops], axis=1)
    rev_full = np.concatenate([zeros, rev_drops], axis=1)
    ks = np.arange(0, H + 1)

    mean_top = top_full.mean(0)
    mean_rev = rev_full.mean(0)
    sem_top = top_full.std(0) / np.sqrt(M)
    sem_rev = rev_full.std(0) / np.sqrt(M)

    auc_top = float(np.trapz(mean_top, ks))
    auc_rev = float(np.trapz(mean_rev, ks))
    auc_signed = auc_top - auc_rev

    return dict(
        ks=ks, mean_top=mean_top, mean_rev=mean_rev,
        sem_top=sem_top, sem_rev=sem_rev,
        auc_top=auc_top, auc_rev=auc_rev, auc_signed=auc_signed,
    )


def plot(results, metrics, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ks = metrics["ks"]
    ax.plot(ks, metrics["mean_top"], color="tab:blue",
            label=f"CRP top-down (AUC={metrics['auc_top']:.3f})")
    ax.fill_between(ks, metrics["mean_top"] - metrics["sem_top"],
                    metrics["mean_top"] + metrics["sem_top"], alpha=0.2, color="tab:blue")
    ax.plot(ks, metrics["mean_rev"], color="tab:red", linestyle="--",
            label=f"CRP bottom-up / reverse (AUC={metrics['auc_rev']:.3f})")
    ax.fill_between(ks, metrics["mean_rev"] - metrics["sem_rev"],
                    metrics["mean_rev"] + metrics["sem_rev"], alpha=0.2, color="tab:red")

    ax.set_xlabel("Concepts knocked out (k)")
    ax.set_ylabel("Mean probability drop")
    ax.set_title(
        f"REVERT knockout — class {results['class_idx']} ({results['class_name']})\n"
        f"mode={results['concept_mode']}+{results['max_target']}, "
        f"layer={results['layer_name']}\n"
        f"AUC_signed (top − rev) = {metrics['auc_signed']:.4f}",
        fontsize=10,
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="REVERT variant: CRP top-down vs CRP bottom-up knockout."
    )
    parser.add_argument("--class-idx", type=int, default=60,
                        help="ImageNet class index (default: 60)")
    parser.add_argument("--layer", default=DEFAULT_LAYER,
                        help=f"Self-attention layer (default: {DEFAULT_LAYER})")
    parser.add_argument("--concept-mode", choices=CONCEPT_MODES, required=True,
                        help="Concept type: head / token / dimension")
    parser.add_argument("--max-target", choices=MAX_TARGETS, required=True,
                        help="Relevance aggregation: sum / max")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    monkey_patch(vision_transformer, verbose=False)
    monkey_patch_zennit(verbose=False)

    results = run(args.class_idx, args.layer, args.concept_mode, args.max_target, device)
    metrics = compute_signed_auc(results["top_drops"], results["rev_drops"])

    base = f"knockout_revert_{args.class_idx}_{args.concept_mode}_{args.max_target}"
    out_png = os.path.join(args.out_dir, f"{base}.png")
    out_npz = os.path.join(args.out_dir, f"{base}.npz")
    out_json = os.path.join(args.out_dir, f"{base}.json")

    plot(results, metrics, out_png)

    np.savez(out_npz,
             top_drops=results["top_drops"],
             rev_drops=results["rev_drops"],
             baseline_probs=results["baseline_probs"])
    print(f"Data saved to {out_npz}")

    summary = {
        "class_idx": results["class_idx"],
        "class_name": results["class_name"],
        "layer_name": results["layer_name"],
        "concept_mode": results["concept_mode"],
        "max_target": results["max_target"],
        "H": results["H"],
        "M": results["M"],
        "auc_top": metrics["auc_top"],
        "auc_rev": metrics["auc_rev"],
        "auc_signed": metrics["auc_signed"],
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics saved to {out_json}")

    print(f"\nAUC_top    = {metrics['auc_top']:.4f}")
    print(f"AUC_rev    = {metrics['auc_rev']:.4f}")
    print(f"AUC_signed = {metrics['auc_signed']:.4f}")


if __name__ == "__main__":
    main()
