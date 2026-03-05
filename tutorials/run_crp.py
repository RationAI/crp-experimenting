import torch
import torch.nn as nn
import numpy as np

import zennit.rules as z_rules
from zennit.composites import LayerMapComposite
from crp.attribution import AttentionAttribution
from lxt.efficient import monkey_patch, monkey_patch_zennit
from torchvision.models import vision_transformer

from utils import set_determinism, load_sample_and_model


def run_attention_aware_crp(model, data, target_class, model_type: str):
    if model_type == "vit":
        monkey_patch(vision_transformer, verbose=True)
    # For non-transformer models, skip monkey_patch - standard Zennit is sufficient
    monkey_patch_zennit(verbose=True)

    composite = LayerMapComposite([
        (nn.Conv2d, z_rules.Gamma(0.25)),
        (nn.Linear, z_rules.Gamma(0.1)),
    ])
    # composite = None

    data.requires_grad = True
    conditions = [{"y": target_class}]

    def one_hot_init(pred):
        mask = torch.zeros_like(pred)
        mask[0, target_class] = 1.0
        return mask

    attributor = AttentionAttribution(model)
    if model_type == "vit":
        def vit_heatmap_modifier(data, on_device=None):
            heatmap = data.grad.detach()
            heatmap = heatmap.to(on_device) if on_device else heatmap
            return torch.sum(heatmap, dim=1)
        attributor.heatmap_modifier = vit_heatmap_modifier

    attr = attributor(data, conditions, composite, mask_map=None, init_rel=one_hot_init)

    relevance = attr.heatmap
    return relevance.detach().cpu().numpy()


def save_relevance(model_type: str, relevance: np.ndarray):
    if model_type == "mnist_linear":
        np.save("relevance_crp.npy", relevance)
    elif model_type == "mnist_conv":
        np.save("relevance_conv_crp.npy", relevance)
    elif model_type == "vgg16":
        np.save("relevance_vgg16_crp.npy", relevance)
    elif model_type == "resnet":
        np.save("relevance_resnet_crp.npy", relevance)
    else:
        np.save("relevance_vit_crp_no_composite.npy", relevance)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run CRP and save relevance for a model.")
    parser.add_argument("--model-type", choices=["mnist_conv", "mnist_linear", "vgg16", "resnet", "vit"], default="vit", help="Model architecture")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_determinism(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data, target_class, model, _ = load_sample_and_model(args.model_type, device)

    print(model)
    
    print(f">>> RUNNING CRP with model '{args.model_type}', target class {target_class}...")
    rel_crp = run_attention_aware_crp(model, data.clone(), target_class, args.model_type)
    save_relevance(args.model_type, rel_crp)
    print("Saved CRP relevance.")


if __name__ == "__main__":
    main()
