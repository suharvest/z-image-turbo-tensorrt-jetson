"""Patch diffusers attention_dispatch.py for Jetson torch 2.5.0 compatibility.

Jetson's torch 2.5.0 build can lack the enable_gqa kwarg, so expand grouped
K/V heads manually before calling attention.
"""
import diffusers.models.attention_dispatch as ad
import torch.nn.functional as F

# Store original _native_attention
_orig_native = ad._native_attention

def _patched_native_attention(**kwargs):
    # Jetson's torch 2.5.0 build can lack the enable_gqa kwarg. Preserve
    # semantics by expanding K/V heads before delegating to diffusers.
    enable_gqa = kwargs.pop("enable_gqa", False)
    if enable_gqa:
        query = kwargs.get("query")
        key = kwargs.get("key")
        value = kwargs.get("value")
        if query is not None and key is not None and value is not None and key.shape[-2] != query.shape[-2]:
            repeats = query.shape[-2] // key.shape[-2]
            kwargs["key"] = key.repeat_interleave(repeats, dim=-2)
            kwargs["value"] = value.repeat_interleave(repeats, dim=-2)
    return _orig_native(**kwargs)

ad._native_attention = _patched_native_attention

# Also patch the direct F.scaled_dot_product_attention call
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

print("Diffusers patched for Jetson torch 2.5.0")
