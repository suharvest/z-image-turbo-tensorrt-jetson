#!/usr/bin/env python3
"""Export Z-Image VAE encoder/decoder ONNX for TensorRT runtime.

The main transformer path needs BF16 for stability, but the VAE is convolutional
and is exported as FP16 to reduce runtime memory and remove PyTorch VAE loading.
"""
import gc
import os

import torch
import torch.nn as nn
from diffusers import AutoencoderKL


MODEL_PATH = os.environ.get("MODEL_PATH", "Tongyi-MAI/Z-Image-Turbo")
MODEL_SUBFOLDER = os.environ.get("VAE_SUBFOLDER", "vae")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "onnx-vae")
RESOLUTION = int(os.environ.get("RESOLUTION", "384"))
EXPORT_DECODER = os.environ.get("EXPORT_DECODER", "1") == "1"
EXPORT_ENCODER = os.environ.get("EXPORT_ENCODER", "1") == "1"
OPSET = int(os.environ.get("OPSET", "17"))

LATENT_H = RESOLUTION // 8
LATENT_W = RESOLUTION // 8


class VAEDecoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent).sample


class VAEEncoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        posterior = self.vae.encode(image).latent_dist
        return posterior.mean, posterior.std


def export_decoder(vae):
    wrapper = VAEDecoderWrapper(vae).eval().cuda()
    latent = torch.randn(1, 16, LATENT_H, LATENT_W, dtype=torch.float16, device="cuda")
    out_path = os.path.join(OUTPUT_DIR, "vae_decoder_fp16.onnx")
    with torch.no_grad():
        sample = wrapper(latent)
    print(f"decoder sample shape={tuple(sample.shape)} dtype={sample.dtype}", flush=True)
    torch.onnx.export(
        wrapper,
        (latent,),
        out_path,
        input_names=["latent"],
        output_names=["image"],
        opset_version=OPSET,
        do_constant_folding=True,
    )
    print(f"exported {out_path}", flush=True)


def export_encoder(vae):
    wrapper = VAEEncoderWrapper(vae).eval().cuda()
    image = torch.randn(1, 3, RESOLUTION, RESOLUTION, dtype=torch.float16, device="cuda")
    out_path = os.path.join(OUTPUT_DIR, "vae_encoder_fp16.onnx")
    with torch.no_grad():
        mean, std = wrapper(image)
    print(
        f"encoder mean shape={tuple(mean.shape)} std shape={tuple(std.shape)} dtype={mean.dtype}",
        flush=True,
    )
    torch.onnx.export(
        wrapper,
        (image,),
        out_path,
        input_names=["image"],
        output_names=["latent_mean", "latent_std"],
        opset_version=OPSET,
        do_constant_folding=True,
    )
    print(f"exported {out_path}", flush=True)


def main():
    if RESOLUTION not in (384, 512):
        raise ValueError(f"RESOLUTION must be 384 or 512, got {RESOLUTION}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading VAE from {MODEL_PATH}/{MODEL_SUBFOLDER}", flush=True)
    vae = AutoencoderKL.from_pretrained(
        MODEL_PATH,
        subfolder=MODEL_SUBFOLDER,
        torch_dtype=torch.float16,
    ).eval().cuda()

    if EXPORT_DECODER:
        export_decoder(vae)
        gc.collect()
        torch.cuda.empty_cache()
    if EXPORT_ENCODER:
        export_encoder(vae)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
