import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random

from prepare_model import get_model, get_data
import zennit.rules as z_rules
from zennit.composites import LayerMapComposite
from lxt.efficient import monkey_patch, monkey_patch_zennit
from torchvision.models import vision_transformer
from torchvision.models.vgg import vgg16_bn
import torchvision.transforms as T
from PIL import Image


def set_determinism(seed: int = 42):
    """Set all random seeds and disable non-deterministic algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Disable cuDNN non-deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sample_and_model(model_type: str, device: torch.device):
    if model_type == "big_conv":
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open("tutorials/images/lizard.jpg")
        data = transform(image).unsqueeze(0).to(device)
        #target_class = 46  # green lizard class
        target_class = 40  # chamele class
        model = vgg16_bn(True).to(device)
        model.eval()
        weights_path = None
    elif model_type == "vit":
        weights = vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
        model = vision_transformer.vit_b_16(weights=weights).to(device)
        model.eval()
        image = Image.open("LRP-eXplains-Transformers/examples/PetImages/lizard.jpg").convert("RGB")
        data = weights.transforms()(image).unsqueeze(0).to(device)
        with torch.no_grad():
            target_class = model(data).argmax().item()
        weights_path = None
    else:
        _, test_data = get_data()
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        data, _ = next(iter(test_loader))
        data = data.to(device)
        model, weights_path = get_model(model_type)
        model.load_state_dict(torch.load(weights_path))
        model.eval()
        model.to(device)
        with torch.no_grad():
            target_class = model(data).argmax().item()
    return data, target_class, model, weights_path


def run_attention_aware_lrp(model, data, target_class, model_type: str):
    # Apply transformer monkey patches (fails hard on error)
    if model_type == "vit":
        monkey_patch(vision_transformer, verbose=True)
    else:
        monkey_patch(model, verbose=True)
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
    if model_type == "linear":
        np.save("relevance_lrp.npy", relevance)
    elif model_type == "conv":
        np.save("relevance_conv_lrp.npy", relevance)
    elif model_type == "big_conv":
        np.save("relevance_big_conv_lrp.npy", relevance)
    else:
        np.save("relevance_vit_lrp.npy", relevance)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run LRP and save relevance for a model.")
    parser.add_argument("--model-type", choices=["conv", "linear", "big_conv", "vit"], default="vit", help="Model architecture")
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
