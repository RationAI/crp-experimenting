# %% [markdown]
# In a backward pass, PyTorch would usually pass the gradients from all neurons of a higher layer to the lower layers and mix them up, however CRP works by passing only specific parts of the gradients and zeroing out the others. Zennit replaces these gradients with actual attribution scores that flow from the last layer to the first one.
#
# In the Image **a)** you see, that we only pass the attributions of the neuron '0' in layer 'layer3' to the next layers. And then again pass only the attributions of neuron '0' and '2' in layer 'layer1' to the input.
# If we were to implement this with a normal PyTorch [backward hook](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_full_backward_hook.html) in layer 'layer1', we would write the following code that sets the input gradient to zero everywhere except for the neurons '0' and '2':

# %%
# %cd ..
# %ls

# %%
import itertools
import types

import torch
import zennit.rules as z_rules
from IPython.display import Image as DisplayImage
from IPython.display import display
from lxt.efficient import monkey_patch, monkey_patch_zennit
from PIL import Image
from torchvision.models import vision_transformer
from zennit.composites import LayerMapComposite
from zennit.image import imgify

from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept


# %%
# 1. Monkey Patch FIRST (This makes the model "Attention-Aware")
monkey_patch(vision_transformer, verbose=True)
monkey_patch_zennit(verbose=True)


# 2. Load Model (Standard PyTorch)
def get_vit_imagenet(device="cuda"):
    weights = vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
    model = vision_transformer.vit_b_16(weights=weights)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model, weights


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this comparison; no GPU detected.")
device = "cuda"
model, weights = get_vit_imagenet(device)

# 3. Load Image
image = Image.open("tutorials/images/lizard.jpg").convert("RGB")
data = weights.transforms()(image).unsqueeze(0).to(device)
data.requires_grad = True

# %%
composite = LayerMapComposite(
    [
        (torch.nn.Conv2d, z_rules.Gamma(0.25)),
        (torch.nn.Linear, z_rules.Gamma(0.1)),
    ]
)

# Define the Concept (The Query)
# ChannelConcept allows us to mask specific channels if we want.
cc = ChannelConcept()

# Initialize the Attributor (The Engine)
# This engine will drive the backward pass.
attributor = CondAttribution(model)

# %%
print(model)

# %% [markdown]
# The `mask` method of the `ChannelConcept` class returns a function similar to the previous mask_hook function.
# `ChannelConcept.mask` allows to mask individual channels as well as MLP neurons per batch.


# %%
# 1. Define the LXT-style modifier function
def chanel_wise_heatmap_modifier(self, data, on_device=None):
    """Computes Relevance = Input * Gradient.
    Does NOT sum over channels (dim 1), preserving RGB info.
    """
    # Get the gradient
    grad = data.grad.detach()

    # Handle device transfer if requested
    if on_device:
        grad = grad.to(on_device)
        data_fixed = data.detach().to(on_device)
    else:
        data_fixed = data.detach()

    relevance = data_fixed * grad

    # Return 4D tensor: (Batch, Channels, Height, Width)
    return relevance


# 2. Monkey Patch your specific instance
# This replaces the 'heatmap_modifier' method ONLY for this 'attributor' object.
attributor.heatmap_modifier = types.MethodType(chanel_wise_heatmap_modifier, attributor)

# %%
prediction = model(data)
target_class = prediction.argmax().item()
print(f"Explaining class: {target_class}")

conditions = [{"y": target_class}]

# ... Run attribution with masking disabled to mirror LRP ...
attr = attributor(data, conditions, composite, mask_map=None)

# 1. Extract the result (It is now RGB!)
# Shape: (1, 3, 224, 224)
rgb_relevance = attr.heatmap[0].detach().cpu()

# 2. Sum channels YOURSELF for visualization (Grayscale)
# Shape: (224, 224)
grayscale_relevance = rgb_relevance.sum(0)
grayscale_relevance = grayscale_relevance / grayscale_relevance.abs().max()

# 3. Save
imgify(grayscale_relevance.unsqueeze(0), symmetric=True, grid=(1, 1)).save(
    "out/lxt_style_result.png"
)

# %%
composite = LayerMapComposite(
    [
        (torch.nn.Conv2d, z_rules.Gamma(0.25)),
        (torch.nn.Linear, z_rules.Gamma(0.1)),
        #     (torch.nn.LayerNorm, z_rules.Pass()),
    ]
)

prediction = model(data)
target_class = prediction.argmax().item()
print(f"Explaining class: {target_class}")

conditions = [{"y": target_class}]

# Run Attribution with masking disabled for LRP equivalence
attr = attributor(data, conditions, composite, mask_map=None)

# Convert to grayscale and normalize
heatmap_crp = attr.heatmap.sum(1)
heatmap_crp = heatmap_crp / heatmap_crp.abs().max()

imgify(heatmap_crp.detach().cpu(), symmetric=True, grid=(1, 1)).save(
    "out/vit_crp_heatmap_norm.png"
)

# %%
lrp_blob = torch.load("LRP-eXplains-Transformers/examples/vit_lrp_heatmap_lizard.pt")
lrp_heatmap = lrp_blob["heatmap"].to(heatmap_crp.device)
lrp_class_idx = lrp_blob.get("class_idx", None)
print(f"LRP saved class idx: {lrp_class_idx}")

# %%
# Token-level gradient heatmap from encoder block output (includes CLS-patch interactions)
# Falls back to the second-last block if the last block gives (near) zero gradients.


def _capture_block_and_grad(block, label):
    captured = {}

    def _hook(_, __, out):
        captured["out"] = out

    handle = block.register_forward_hook(_hook)
    pred = model(data.requires_grad_())
    handle.remove()
    if "out" not in captured:
        raise RuntimeError(f"Did not capture output for {label}")
    out = captured["out"]
    print(f"Captured '{label}' output shape: {tuple(out.shape)}")
    target_logit = pred[0, target_class]
    grad = torch.autograd.grad(
        target_logit, out, allow_unused=False, retain_graph=True
    )[0]
    if grad is None:
        raise RuntimeError(f"Gradient wrt '{label}' is None")
    print(
        f"Gradient wrt '{label}' shape: {tuple(grad.shape)}, max={grad.abs().max().item():.3e}"
    )
    return grad


# Try last block output first
layer_label = "encoder.layers[-1]"
last_block = model.encoder.layers[-1]
grad_single = _capture_block_and_grad(last_block, layer_label)

# If nearly zero, fall back to second-last block
if grad_single.abs().max() < 1e-8 and len(model.encoder.layers) > 1:
    layer_label = "encoder.layers[-2]"
    snd_block = model.encoder.layers[-2]
    grad_single = _capture_block_and_grad(snd_block, layer_label)

# Build token heatmap
if grad_single.dim() == 3:
    heatmap_single = grad_single.sum(2)
    # Remove CLS token if present
    if heatmap_single.size(1) == 197:
        heatmap_single = heatmap_single[:, 1:]
    elif heatmap_single.size(1) == 196:
        pass
    else:
        raise RuntimeError(
            f"Unexpected token count: {heatmap_single.size(1)}. Expected 196 or 197."
        )
    h = w = int(heatmap_single.size(1) ** 0.5)
    if h * w != heatmap_single.size(1):
        raise RuntimeError(
            f"Cannot reshape to square grid: {heatmap_single.size(1)} tokens."
        )
    heatmap_single = heatmap_single.reshape(-1, h, w)
elif grad_single.dim() == 2:
    raise RuntimeError(
        "Got only 2D gradient (batch, 768). Hook an earlier layer that feeds CLS via attention."
    )
else:
    raise RuntimeError(f"Unexpected grad shape: {grad_single.shape}")

# Safe normalization
eps = 1e-6
denom = heatmap_single.abs().max()
heatmap_single = heatmap_single / (denom + eps)

torch.save(
    {
        "heatmap": heatmap_single.detach().cpu(),
        "layer": layer_label,
        "class_idx": target_class,
    },
    "out/vit_crp_heatmap_tokens.pt",
)

imgify(heatmap_single.unsqueeze(1).detach().cpu(), symmetric=True, grid=(1, 1)).save(
    "out/vit_crp_heatmap_tokens.png"
)

# %%
# Check raw heatmaps before normalization
print("=== LRP Heatmap Stats ===")
print(f"Min: {lrp_heatmap.min().item():.6e}, Max: {lrp_heatmap.max().item():.6e}")
print(f"Mean: {lrp_heatmap.mean().item():.6e}, Std: {lrp_heatmap.std().item():.6e}")
print(f"Sum positive: {lrp_heatmap[lrp_heatmap > 0].sum().item():.6e}")
print(f"Sum negative: {lrp_heatmap[lrp_heatmap < 0].sum().item():.6e}")

print("\n=== CRP Heatmap Stats ===")
print(f"Min: {heatmap_crp.min().item():.6e}, Max: {heatmap_crp.max().item():.6e}")
print(f"Mean: {heatmap_crp.mean().item():.6e}, Std: {heatmap_crp.std().item():.6e}")
print(f"Sum positive: {heatmap_crp[heatmap_crp > 0].sum().item():.6e}")
print(f"Sum negative: {heatmap_crp[heatmap_crp < 0].sum().item():.6e}")

print("\n=== Difference Stats ===")
diff = heatmap_crp - lrp_heatmap
print(f"Diff min: {diff.min().item():.6e}, Diff max: {diff.max().item():.6e}")
print(f"Diff mean: {diff.mean().item():.6e}")
print(f"Sign flips: {((lrp_heatmap > 0) != (heatmap_crp > 0)).sum().item()}")

# %%
diff_map_crp = heatmap_crp - lrp_heatmap

print(f"max diff (crp vs lrp): {diff_map_crp.abs().max().item():.6e}")
print(f"mean diff (crp vs lrp): {diff_map_crp.abs().mean().item():.6e}")
if lrp_class_idx is not None:
    print(f"LRP heatmap class idx: {lrp_class_idx}")
    print(f"CRP heatmap class idx: {target_class}")

imgify(diff_map_crp.detach().cpu(), symmetric=True, grid=(1, 1)).save(
    "out/diff_map_crp.png"
)

# %%
print(diff_map_crp.type)
imgify(diff_map_crp.detach().cpu(), symmetric=True, grid=(1, 1))

# %%
display(DisplayImage("out/lxt_style_result.png"))

display(DisplayImage("out/vit_crp_heatmap_pass_norm.png"))

display(DisplayImage("out/diff_map.png"))
display(DisplayImage("out/diff_map_crp.png"))

# %%
display(image)
display(DisplayImage("out/vit_crp_heatmap.png"))

# %%
display(DisplayImage("vit_crp_heatmap.png"))

# %%
model, weights = get_vit_imagenet()

cc = ChannelConcept()
attributor = CondAttribution(model)

prediction = model(data)
# target_class = prediction.argmax().item()
classes = torch.topk(prediction, 5)
# the best class
target_class = classes.indices[0, 0].item()
# the second best calss
# target_class = classes.indices[0, 1].item()
# the fifth best calss
# target_class = classes.indices[0, 4].item()

class_name = weights.meta["categories"][target_class]

print(f"Explaining class: {target_class} ('{class_name}')")
conditions = [{"y": target_class}]

heatmap_list = []

for conv_gamma, lin_gamma in itertools.product(
    [0.1, 0.25, 100], [0, 0.01, 0.05, 0.1, 1]
):
    composite = LayerMapComposite(
        [
            (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
            (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
        ]
    )

    # Run Attribution
    attr = attributor(data, conditions, composite, mask_map=cc.mask)

    heatmap_list.append(attr.heatmap.detach().cpu())

# Join
# Result shape: (15, 1, 3, 224, 224)
stacked_heatmaps = torch.stack(heatmap_list)

# Fix Dimensions if necessary
if stacked_heatmaps.ndim == 5:
    stacked_heatmaps = stacked_heatmaps.squeeze(1)

# Convert to Grayscale
final_tensor = stacked_heatmaps.sum(1)

print(f"Final tensor shape: {final_tensor.shape}")
imgify(final_tensor, symmetric=True, grid=(3, 5)).save(
    "out/vit_crp_heatmap_chameleon.png"
)

# %%
display(DisplayImage("vit_crp_heatmap_lizard.png"))

# %%
display(DisplayImage("vit_crp_heatmap_chameleon.png"))

# %%
display(DisplayImage("vit_lrp_heatmap_chameleon.png"))
