#!/usr/bin/env python3
"""Convert drbaph/Z-Image-Turbo-FP8 ComfyUI weights to Diffusers layout.

The important detail is that this preserves torch.float8_e4m3fn tensors.
Do not upcast the transformer to fp16 on a 16 GB Jetson Orin NX.
"""

import argparse
import gc
import json
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def tensor_nbytes(tensor):
    return tensor.numel() * tensor.element_size()


def path_size(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total


def fmt_bytes(value):
    return f"{value / (1024 ** 3):.2f} GiB"


def map_key(key):
    if key.startswith("final_layer."):
        return "all_final_layer.2-1." + key[len("final_layer.") :]
    if key.startswith("x_embedder."):
        return "all_x_embedder.2-1." + key[len("x_embedder.") :]
    if key.startswith("t_embedder."):
        return "all_t_embedder.2-1." + key[len("t_embedder.") :]
    if "attention.qkv.weight" in key:
        return key.replace("attention.qkv.weight", "__QKV_SPLIT__")
    if "attention.k_norm" in key:
        return key.replace("attention.k_norm", "attention.norm_k")
    if "attention.q_norm" in key:
        return key.replace("attention.q_norm", "attention.norm_q")
    if "attention.out.weight" in key:
        return key.replace("attention.out.weight", "attention.to_out.0.weight")
    return key


def copy_component(src, dst, mode):
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    if mode == "symlink":
        os.symlink(src, dst, target_is_directory=src.is_dir())
    elif src.is_dir():
        shutil.copytree(src, dst, symlinks=True)
    else:
        shutil.copy2(src, dst)


def flush_shard(shard, shard_sizes, weight_map, transformer_dir, shard_idx, total_shards_hint):
    if not shard:
        return shard_idx
    filename = f"diffusion_pytorch_model-{shard_idx:05d}-of-{total_shards_hint:05d}.safetensors"
    out_path = transformer_dir / filename
    save_file(shard, out_path)
    actual_size = out_path.stat().st_size
    shard_sizes.append((filename, actual_size, len(shard)))
    for key in shard:
        weight_map[key] = filename
    print(f"wrote {filename}: {len(shard)} tensors, {fmt_bytes(actual_size)}")
    shard.clear()
    gc.collect()
    return shard_idx + 1


def rewrite_index_filenames(transformer_dir, shard_sizes, weight_map, metadata):
    total = len(shard_sizes)
    renamed_map = {}

    for idx, (old_name, _size, _count) in enumerate(shard_sizes, start=1):
        new_name = f"diffusion_pytorch_model-{idx:05d}-of-{total:05d}.safetensors"
        if old_name != new_name:
            (transformer_dir / old_name).rename(transformer_dir / new_name)

    for key, old_name in weight_map.items():
        old_idx = int(old_name.split("-")[1])
        renamed_map[key] = f"diffusion_pytorch_model-{old_idx:05d}-of-{total:05d}.safetensors"

    index = {"metadata": metadata, "weight_map": renamed_map}
    with open(transformer_dir / "diffusion_pytorch_model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description="Preserve-FP8 Z-Image-Turbo conversion")
    parser.add_argument("--src_base", default="/models/z-image-turbo", help="Diffusers base model with configs, T5, VAE, tokenizer")
    parser.add_argument("--src_fp8", default="/models/z-image-turbo-fp8/z_image_turbo_fp8_e4m3fn.safetensors")
    parser.add_argument("--dst", default="/models/z-image-turbo-fp8-diffusers")
    parser.add_argument("--max_shard_size_gb", type=float, default=2.25)
    parser.add_argument("--copy_mode", choices=["symlink", "copy"], default="symlink")
    args = parser.parse_args()

    src_base = Path(args.src_base)
    src_fp8 = Path(args.src_fp8)
    dst = Path(args.dst)
    transformer_dir = dst / "transformer"
    max_shard_bytes = int(args.max_shard_size_gb * 1024**3)

    if not src_fp8.exists():
        raise FileNotFoundError(src_fp8)
    if not (src_base / "transformer" / "config.json").exists():
        raise FileNotFoundError(src_base / "transformer" / "config.json")

    print("Input layout")
    print(f"  base model: {src_base}")
    print(f"  fp8 checkpoint: {src_fp8} ({fmt_bytes(src_fp8.stat().st_size)})")
    print(f"  output: {dst}")
    print(f"  shard target: {fmt_bytes(max_shard_bytes)}")
    print()
    print("Memory comparison")
    print(f"  source fp8 transformer on disk: {fmt_bytes(src_fp8.stat().st_size)}")
    print(f"  estimated fp16 transformer if upcast: {fmt_bytes(src_fp8.stat().st_size * 2)}")
    print(f"  expected fp8 diffusers transformer: about {fmt_bytes(src_fp8.stat().st_size)} plus index overhead")
    print()

    transformer_dir.mkdir(parents=True, exist_ok=True)
    for old in transformer_dir.glob("*.safetensors"):
        old.unlink()

    with safe_open(src_fp8, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        source_tensor_bytes = sum(f.get_tensor(k).numel() * f.get_tensor(k).element_size() for k in keys)

    total_shards_hint = max(1, int(source_tensor_bytes / max_shard_bytes) + 2)
    print(f"Converting {len(keys)} source tensors, raw tensor bytes {fmt_bytes(source_tensor_bytes)}")
    print(f"Expected shard count: about {total_shards_hint - 1} to {total_shards_hint}")

    shard = OrderedDict()
    shard_bytes = 0
    shard_idx = 1
    shard_sizes = []
    weight_map = {}
    dtype_counts = {}
    tensor_count = 0
    out_tensor_bytes = 0

    with safe_open(src_fp8, framework="pt", device="cpu") as f:
        for key in keys:
            mapped = map_key(key)
            tensor = f.get_tensor(key)
            dtype_counts[str(tensor.dtype)] = dtype_counts.get(str(tensor.dtype), 0) + 1

            out_items = []
            if "__QKV_SPLIT__" in mapped:
                base = mapped.replace("__QKV_SPLIT__", "")
                q, k, v = tensor.chunk(3, dim=0)
                out_items.extend(
                    [
                        (base + "attention.to_q.weight", q.contiguous()),
                        (base + "attention.to_k.weight", k.contiguous()),
                        (base + "attention.to_v.weight", v.contiguous()),
                    ]
                )
            else:
                out_items.append((mapped, tensor))

            for out_key, out_tensor in out_items:
                nbytes = tensor_nbytes(out_tensor)
                if shard and shard_bytes + nbytes > max_shard_bytes:
                    shard_idx = flush_shard(shard, shard_sizes, weight_map, transformer_dir, shard_idx, total_shards_hint)
                    shard_bytes = 0

                shard[out_key] = out_tensor
                shard_bytes += nbytes
                out_tensor_bytes += nbytes
                tensor_count += 1

            del tensor

    flush_shard(shard, shard_sizes, weight_map, transformer_dir, shard_idx, total_shards_hint)

    metadata = {"total_size": str(out_tensor_bytes), "format": "pt"}
    rewrite_index_filenames(transformer_dir, shard_sizes, weight_map, metadata)

    shutil.copy2(src_base / "transformer" / "config.json", transformer_dir / "config.json")

    for name in ["text_encoder", "tokenizer", "tokenizer_2", "vae", "scheduler", "feature_extractor", "image_processor"]:
        src = src_base / name
        if src.exists():
            copy_component(src, dst / name, args.copy_mode)
            print(f"{args.copy_mode}: {name}")

    for name in ["model_index.json", "assets"]:
        src = src_base / name
        if src.exists():
            copy_component(src, dst / name, args.copy_mode)
            print(f"{args.copy_mode}: {name}")

    final_transformer_size = path_size(transformer_dir)
    print()
    print("Conversion complete")
    print(f"  output tensors: {tensor_count}")
    print(f"  source dtypes: {dtype_counts}")
    print(f"  transformer output size: {fmt_bytes(final_transformer_size)}")
    print("  shard sizes:")
    for filename in sorted(p.name for p in transformer_dir.glob("*.safetensors")):
        print(f"    {filename}: {fmt_bytes((transformer_dir / filename).stat().st_size)}")
    print()
    print("Before/after memory comparison")
    print(f"  old fp16 transformer path: about {fmt_bytes(out_tensor_bytes * 2)} resident weights")
    print(f"  new fp8 transformer path:  about {fmt_bytes(out_tensor_bytes)} resident weights")
    print("  T5 is intended to be loaded only for prompt encoding, then released before denoising.")


if __name__ == "__main__":
    main()
