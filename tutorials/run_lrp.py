import torch
import torch.nn as nn
import numpy as np

import zennit.rules as z_rules
from zennit.composites import LayerMapComposite
from lxt.efficient import monkey_patch, monkey_patch_zennit
from torchvision.models import vision_transformer

from utils import set_determinism, load_sample_and_model


def run_attention_aware_lrp(model, data, target_class, model_type: str):
    # Apply transformer monkey patches (fails hard on error)
    if model_type == "vit":
        monkey_patch(vision_transformer, verbose=True)
    # For non-transformer models, skip monkey_patch - standard Zennit is sufficient
    monkey_patch_zennit(verbose=True)

    composite = LayerMapComposite([
        (nn.Conv2d, z_rules.Gamma(0.25)),
        (nn.Linear, z_rules.Gamma(0.1)),
    ])

    data.requires_grad_(True)
    composite.register(model)
    y = model(data)
    y[0, target_class].backward()
    composite.remove()

    relevance = data.grad
    if relevance.ndim == 4:
        relevance = relevance.sum(1)
    return relevance.detach().cpu().numpy()


def save_relevance(model_type: str, relevance: np.ndarray):
    if model_type == "mnist_linear":
        np.save("relevance_lrp.npy", relevance)
    elif model_type == "mnist_conv":
        np.save("relevance_conv_lrp.npy", relevance)
    elif model_type == "vgg16":
        np.save("relevance_vgg16_lrp.npy", relevance)
    elif model_type == "resnet":
        np.save("relevance_resnet_lrp.npy", relevance)
    else:
        np.save("relevance_vit_lrp.npy", relevance)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run LRP and save relevance for a model.")
    parser.add_argument("--model-type", choices=["mnist_conv", "mnist_linear", "vgg16", "resnet", "vit"], default="vit", help="Model architecture")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_determinism(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data, target_class, model, _ = load_sample_and_model(args.model_type, device)


    print(model)
    
    print(f">>> RUNNING LRP with model '{args.model_type}', target class {target_class}...")
    rel_lrp = run_attention_aware_lrp(model, data.clone(), target_class, args.model_type)
    save_relevance(args.model_type, rel_lrp)
    print("Saved LRP relevance.")


if __name__ == "__main__":
    main()
