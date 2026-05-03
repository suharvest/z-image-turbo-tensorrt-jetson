#!/usr/bin/env python3
"""Z-Image-Turbo inference script."""

import torch
from diffusers import ZImagePipeline
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Run Z-Image-Turbo inference")
    parser.add_argument("--prompt", type=str, default="A cute cat")
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/models/z-image-turbo"))
    parser.add_argument("--output", type=str, default="/output")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading model from {args.model}...")
    pipe = ZImagePipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to("cuda")

    # Enable flash attention if available
    try:
        pipe.transformer.set_attention_backend("flash")
    except Exception:
        try:
            pipe.transformer.set_attention_backend("_flash_3")
        except Exception:
            print("Using SDPA (default)")

    print(f"Generating: {args.prompt}")
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).images[0]

    output_path = os.path.join(args.output, "output.png")
    image.save(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
