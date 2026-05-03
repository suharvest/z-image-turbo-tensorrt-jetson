#!/usr/bin/env python3
"""Diagnostic Z-Image-Turbo inference for Jetson torch 2.5.0.

Runs the converted ComfyUI FP8 checkpoint through Diffusers and reports the
first non-finite tensor seen in latents, AdaLN, attention, MLP, or VAE decode.
"""

import argparse
import os
import sys

import torch


def patch_jetson_torch():
    import torch.distributed as dist
    import torch.nn.functional as F

    if not hasattr(dist, "device_mesh"):
        dist.device_mesh = type("device_mesh", (), {"DeviceMesh": type("FakeDM", (), {})})

    try:
        import torch._dynamo.utils as du

        if not hasattr(du, "NP_SUPPORTED_MODULES"):
            du.NP_SUPPORTED_MODULES = {}
    except Exception:
        pass

    orig_sdpa = F.scaled_dot_product_attention

    def patched_sdpa(*args, **kwargs):
        enable_gqa = kwargs.pop("enable_gqa", False)
        if enable_gqa and len(args) >= 3:
            query, key, value = args[:3]
            if query.ndim >= 3 and key.shape[-3] != query.shape[-3]:
                repeats = query.shape[-3] // key.shape[-3]
                key = key.repeat_interleave(repeats, dim=-3)
                value = value.repeat_interleave(repeats, dim=-3)
                args = (query, key, value, *args[3:])
        return orig_sdpa(*args, **kwargs)

    F.scaled_dot_product_attention = patched_sdpa

    try:
        import diffusers.models.attention_dispatch as ad

        orig_native = ad._native_attention

        def patched_native_attention(**kwargs):
            enable_gqa = kwargs.pop("enable_gqa", False)
            if enable_gqa:
                query = kwargs.get("query")
                key = kwargs.get("key")
                value = kwargs.get("value")
                if query is not None and key is not None and value is not None and key.shape[-2] != query.shape[-2]:
                    repeats = query.shape[-2] // key.shape[-2]
                    kwargs["key"] = key.repeat_interleave(repeats, dim=-2)
                    kwargs["value"] = value.repeat_interleave(repeats, dim=-2)
            return orig_native(**kwargs)

        ad._native_attention = patched_native_attention
    except Exception:
        pass


def iter_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensors(item)


def tensor_stats(tensor):
    data = tensor.detach()
    finite = torch.isfinite(data)
    nan_count = torch.isnan(data).sum().item()
    inf_count = torch.isinf(data).sum().item()
    if finite.any().item():
        finite_data = data[finite].float()
        return {
            "shape": tuple(data.shape),
            "dtype": str(data.dtype),
            "nan": nan_count,
            "inf": inf_count,
            "min": finite_data.min().item(),
            "max": finite_data.max().item(),
            "mean": finite_data.mean().item(),
        }
    return {
        "shape": tuple(data.shape),
        "dtype": str(data.dtype),
        "nan": nan_count,
        "inf": inf_count,
        "min": float("nan"),
        "max": float("nan"),
        "mean": float("nan"),
    }


def print_stats(label, value):
    tensors = list(iter_tensors(value))
    if not tensors:
        return
    stats = tensor_stats(tensors[0])
    print(
        f"{label}: shape={stats['shape']} dtype={stats['dtype']} "
        f"nan={stats['nan']} inf={stats['inf']} "
        f"min={stats['min']:.6g} max={stats['max']:.6g} mean={stats['mean']:.6g}",
        flush=True,
    )


def install_nan_hooks(pipe):
    state = {"step": -1, "first": None}

    def transformer_pre_hook(_module, args):
        state["step"] += 1
        if args:
            print_stats(f"step={state['step']} transformer_input", args[0])

    pipe.transformer.register_forward_pre_hook(transformer_pre_hook)

    def should_hook(name):
        return (
            "adaLN_modulation" in name
            or ".attention" in name
            or ".feed_forward" in name
            or name.endswith("final_layer")
            or name.startswith("vae.")
        )

    def make_hook(name):
        def hook(_module, _inputs, output):
            if state["first"] is not None:
                return
            for tensor in iter_tensors(output):
                if not torch.isfinite(tensor).all().item():
                    state["first"] = (state["step"], name, tensor_stats(tensor))
                    step, module_name, stats = state["first"]
                    print(
                        f"FIRST_NONFINITE step={step} module={module_name} "
                        f"shape={stats['shape']} dtype={stats['dtype']} "
                        f"nan={stats['nan']} inf={stats['inf']} "
                        f"finite_min={stats['min']:.6g} finite_max={stats['max']:.6g} "
                        f"finite_mean={stats['mean']:.6g}",
                        flush=True,
                    )
                    raise RuntimeError(f"non-finite tensor at step={step} module={module_name}")

        return hook

    for name, module in pipe.named_modules():
        if should_hook(name):
            module.register_forward_hook(make_hook(name))

    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/z-image-turbo-fp8-diffusers")
    parser.add_argument("--prompt", default="A cute orange tabby cat sitting on a sunny windowsill, digital art")
    parser.add_argument("--output", default="/output/output.png")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = parser.parse_args()

    patch_jetson_torch()

    if not torch.cuda.is_available():
        print("CUDA is required for this diagnostic.", file=sys.stderr)
        sys.exit(2)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    print(f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    print(f"device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}", flush=True)
    print(f"loading model={args.model} dtype={dtype}", flush=True)

    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="balanced",
        max_memory={0: "10GB"},
    )
    state = install_nan_hooks(pipe)

    step_latents = []

    def callback(_pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        print_stats(f"step={step} scheduler_latents timestep={int(timestep.item())}", latents)
        step_latents.append(latents.detach().float().cpu())
        return callback_kwargs

    result = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(args.seed),
        output_type="np",
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )

    image = result.images[0]
    import numpy as np
    from PIL import Image

    print(
        f"image: shape={image.shape} nan={np.isnan(image).sum()} "
        f"inf={np.isinf(image).sum()} min={np.nanmin(image):.6g} "
        f"max={np.nanmax(image):.6g} mean={np.nanmean(image):.6g}",
        flush=True,
    )
    if state["first"] is None:
        print("FIRST_NONFINITE none", flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    clean = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    Image.fromarray((np.clip(clean, 0, 1) * 255).astype(np.uint8)).save(args.output)
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
