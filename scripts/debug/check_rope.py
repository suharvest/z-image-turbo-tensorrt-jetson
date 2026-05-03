"""Inspect RoPE application in Z-Image transformer."""
import inspect

# Monkey-patches first
import torch.distributed as dist
if not hasattr(dist, "device_mesh"):
    dist.device_mesh = type("_", (), {"DeviceMesh": type("_", (), {})})
import torch._dynamo.utils as du
if not hasattr(du, "NP_SUPPORTED_MODULES"):
    du.NP_SUPPORTED_MODULES = {}

from diffusers.models.transformers.transformer_z_image import apply_rotary_emb, ZImageAttnProcessor2_0

print("=== apply_rotary_emb ===")
print(inspect.getsource(apply_rotary_emb))

print("\n=== ZImageAttnProcessor2_0.__call__ ===")
print(inspect.getsource(ZImageAttnProcessor2_0.__call__))
