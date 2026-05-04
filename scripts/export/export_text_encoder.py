#!/usr/bin/env python3
"""Export Z-Image Qwen3 text encoder ONNX for TensorRT runtime."""
import os

import torch
import torch.nn as nn

import torch.distributed as dist

if not hasattr(dist, "device_mesh"):
    dist.device_mesh = type("device_mesh", (), {"DeviceMesh": type("FakeDM", (), {})})
try:
    import torch._dynamo.utils as dynamo_utils

    if not hasattr(dynamo_utils, "NP_SUPPORTED_MODULES"):
        dynamo_utils.NP_SUPPORTED_MODULES = {}
except Exception:
    pass

from transformers import Qwen3Model


MODEL_PATH = os.environ.get("MODEL_PATH", "Tongyi-MAI/Z-Image-Turbo")
TEXT_ENCODER_SUBFOLDER = os.environ.get("TEXT_ENCODER_SUBFOLDER", "text_encoder")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "onnx-text-encoder")
TEXT_TOKENS = int(os.environ.get("TEXT_TOKENS", "128"))
OPSET = int(os.environ.get("OPSET", "17"))
DTYPE = os.environ.get("DTYPE", "bf16").lower()


class TextEncoderWrapper(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids, attention_mask):
        out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return out.hidden_states[-2]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if DTYPE == "fp16":
        torch_dtype = torch.float16
        suffix = "fp16"
    elif DTYPE == "bf16":
        torch_dtype = torch.bfloat16
        suffix = "bf16"
    else:
        raise ValueError(f"Unsupported DTYPE={DTYPE}. Use bf16 or fp16.")

    print(f"Loading text encoder from {MODEL_PATH}/{TEXT_ENCODER_SUBFOLDER}", flush=True)
    load_kwargs = {
        "subfolder": TEXT_ENCODER_SUBFOLDER,
        "torch_dtype": torch_dtype,
    }
    try:
        text_encoder = Qwen3Model.from_pretrained(
            MODEL_PATH,
            attn_implementation="eager",
            **load_kwargs,
        )
    except TypeError:
        text_encoder = Qwen3Model.from_pretrained(MODEL_PATH, **load_kwargs)
    text_encoder = text_encoder.eval().cuda()
    wrapper = TextEncoderWrapper(text_encoder).eval().cuda()

    input_ids = torch.ones(1, TEXT_TOKENS, dtype=torch.long, device="cuda")
    attention_mask = torch.ones(1, TEXT_TOKENS, dtype=torch.long, device="cuda")
    with torch.no_grad():
        sample = wrapper(input_ids, attention_mask)
    print(f"text encoder sample shape={tuple(sample.shape)} dtype={sample.dtype}", flush=True)

    out_path = os.path.join(OUTPUT_DIR, f"text_encoder_{suffix}.onnx")
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["prompt_embeds"],
        opset_version=OPSET,
        do_constant_folding=True,
    )
    print(f"exported {out_path}", flush=True)


if __name__ == "__main__":
    main()
