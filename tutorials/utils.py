import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import torchvision.transforms as T
from PIL import Image
from torchvision.models import vision_transformer
from torchvision.models.vgg import vgg16_bn
from torchvision.models import resnet50

from prepare_model import get_model, get_data


def set_determinism(seed: int = 42):
    """Set all random seeds and disable non-deterministic algorithms."""

    # Necessary to prevent non-determinism in GPU computations in zennit monkey patched backpropagation.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Disable cuDNN non-deterministic algorithms
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sample_and_model(model_type: str, device: torch.device):
    if model_type == "vgg16":
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open("tutorials/lizard.jpg")
        data = transform(image).unsqueeze(0).to(device)
        #target_class = 46  # green lizard class
        target_class = 40  # chameleon class
        model = vgg16_bn(True).to(device)
        model.eval()
        weights_path = None
    elif model_type == "resnet":
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open("tutorials/lizard.jpg")
        data = transform(image).unsqueeze(0).to(device)
        target_class = 40  # chameleon class
        model = resnet50(pretrained=True).to(device)
        model.eval()
        weights_path = None
    elif model_type == "vit":
        weights = vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
        model = vision_transformer.vit_b_16(weights=weights).to(device)
        model.eval()
        image = Image.open("tutorials/lizard.jpg").convert("RGB")
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
