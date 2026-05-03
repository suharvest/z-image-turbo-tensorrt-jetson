"""Z-Image-Turbo image generation on Jetson Orin with fp8 model.
Must be run AFTER numpy is fixed to 1.x.
"""
import os, torch

# Monkey-patch Jetson torch missing device_mesh
import torch.distributed as dist
if not hasattr(dist, "device_mesh"):
    dist.device_mesh = type("device_mesh", (), {"DeviceMesh": type("FakeDM", (), {})})

# Monkey-patch missing NP_SUPPORTED_MODULES
import torch._dynamo.utils as du
if not hasattr(du, "NP_SUPPORTED_MODULES"):
    du.NP_SUPPORTED_MODULES = {}

# Verify CUDA
print(f"torch: {torch.__version__}  cuda: {torch.cuda.is_available()}", flush=True)
print(f"arch: {torch.cuda.get_arch_list()}", flush=True)
a = torch.tensor([1.0, 2.0]).cuda()
print(f"CUDA test: {a * 2}", flush=True)

# Patch diffusers SDPA for Jetson torch 2.5.0 builds without enable_gqa.
import torch.nn.functional as F
_orig_sdpa = F.scaled_dot_product_attention
def _patched_sdpa(*args, **kwargs):
    enable_gqa = kwargs.pop("enable_gqa", False)
    if enable_gqa and len(args) >= 3:
        query, key, value = args[:3]
        if query.ndim >= 3 and key.shape[-3] != query.shape[-3]:
            repeats = query.shape[-3] // key.shape[-3]
            key = key.repeat_interleave(repeats, dim=-3)
            value = value.repeat_interleave(repeats, dim=-3)
            args = (query, key, value, *args[3:])
    return _orig_sdpa(*args, **kwargs)
F.scaled_dot_product_attention = _patched_sdpa

# Also set attention to use SDPA which is most stable
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# Generate
from diffusers import ZImagePipeline

print("Loading model...", flush=True)
pipe = ZImagePipeline.from_pretrained(
    "/models/z-image-turbo-fp8-diffusers",
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    max_memory={0: "10GB"},
)
print("Generating...", flush=True)
image = pipe(
    prompt="A cute orange tabby cat sitting on a sunny windowsill, soft natural lighting, photorealistic, high detail",
    height=512,
    width=512,
    num_inference_steps=8,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
    output_type="np",
).images[0]

# Check for NaN and clip
import numpy as np
print(f"Image stats: min={image.min():.4f} max={image.max():.4f} mean={image.mean():.4f}", flush=True)
nan_count = np.isnan(image).sum()
if nan_count > 0:
    print(f"WARNING: {nan_count} NaN values, clipping...", flush=True)
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

# Clamp to valid range
image = np.clip(image, 0, 1)
from PIL import Image as PILImage
img = PILImage.fromarray((image * 255).astype(np.uint8))
# Clamp to valid range
image = np.clip(image, 0, 1)
from PIL import Image as PILImage
img = PILImage.fromarray((image * 255).astype(np.uint8))
os.makedirs("/output", exist_ok=True)
img.save("/output/output.png")
print("IMAGE SAVED!", flush=True)
