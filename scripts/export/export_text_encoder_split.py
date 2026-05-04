#!/usr/bin/env python3
"""Export Z-Image Qwen3 text encoder as smaller ONNX layer groups.

The Z-Image pipeline consumes ``output.hidden_states[-2]`` from Qwen3Model:
the hidden state after the final decoder layer, before the model's final norm.
This exporter preserves that boundary while splitting the 36-layer text encoder
into TensorRT-buildable chunks for Jetson.
"""
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
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


MODEL_PATH = os.environ.get("MODEL_PATH", "Tongyi-MAI/Z-Image-Turbo")
TEXT_ENCODER_SUBFOLDER = os.environ.get("TEXT_ENCODER_SUBFOLDER", "text_encoder")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "onnx-text-encoder-split")
TEXT_TOKENS = int(os.environ.get("TEXT_TOKENS", "128"))
OPSET = int(os.environ.get("OPSET", "17"))
DTYPE = os.environ.get("DTYPE", "bf16").lower()
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "4"))


class TextEncoderGroupWrapper(nn.Module):
    def __init__(self, text_encoder, start_layer, end_layer):
        super().__init__()
        self.text_encoder = text_encoder
        self.start_layer = start_layer
        self.end_layer = end_layer

    def _causal_masks(self, hidden_states, attention_mask, cache_position, position_ids):
        mask_kwargs = {
            "config": self.text_encoder.config,
            "input_embeds": hidden_states,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if self.text_encoder.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        return causal_mask_mapping

    def forward(self, input_ids, attention_mask, hidden_states=None):
        if self.start_layer == 0:
            hidden_states = self.text_encoder.embed_tokens(input_ids)
        cache_position = torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0)
        causal_mask_mapping = self._causal_masks(
            hidden_states,
            attention_mask,
            cache_position,
            position_ids,
        )
        position_embeddings = self.text_encoder.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.text_encoder.layers[self.start_layer : self.end_layer + 1]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        return hidden_states


def _torch_dtype():
    if DTYPE == "fp16":
        return torch.float16, "fp16"
    if DTYPE == "bf16":
        return torch.bfloat16, "bf16"
    raise ValueError(f"Unsupported DTYPE={DTYPE}. Use bf16 or fp16.")


def _group_ranges(num_layers):
    ranges_env = os.environ.get("TEXT_ENCODER_GROUPS", "").strip()
    if ranges_env:
        ranges = []
        for item in ranges_env.split(","):
            start, end = item.split("-")
            ranges.append((int(start), int(end)))
        return ranges
    return [(start, min(start + GROUP_SIZE - 1, num_layers - 1)) for start in range(0, num_layers, GROUP_SIZE)]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch_dtype, suffix = _torch_dtype()

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

    num_layers = int(text_encoder.config.num_hidden_layers)
    input_ids = torch.ones(1, TEXT_TOKENS, dtype=torch.long, device="cuda")
    attention_mask = torch.ones(1, TEXT_TOKENS, dtype=torch.long, device="cuda")
    hidden_states = torch.zeros(
        1,
        TEXT_TOKENS,
        int(text_encoder.config.hidden_size),
        dtype=torch_dtype,
        device="cuda",
    )

    for start, end in _group_ranges(num_layers):
        if start < 0 or end >= num_layers or start > end:
            raise ValueError(f"Invalid group range {start}-{end} for {num_layers} layers")
        wrapper = TextEncoderGroupWrapper(text_encoder, start, end).eval().cuda()
        base_name = f"text_encoder_group_{start:02d}_{end:02d}_{suffix}"
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}.onnx")
        print(f"Exporting {base_name}", flush=True)
        with torch.no_grad():
            if start == 0:
                sample = wrapper(input_ids, attention_mask)
                args = (input_ids, attention_mask)
                input_names = ["input_ids", "attention_mask"]
            else:
                sample = wrapper(input_ids, attention_mask, hidden_states)
                args = (input_ids, attention_mask, hidden_states)
                input_names = ["input_ids", "attention_mask", "hidden_states"]
        print(f"  sample shape={tuple(sample.shape)} dtype={sample.dtype}", flush=True)
        torch.onnx.export(
            wrapper,
            args,
            out_path,
            input_names=input_names,
            output_names=["hidden_states"],
            opset_version=OPSET,
            do_constant_folding=True,
        )
        print(f"  exported {out_path}", flush=True)


if __name__ == "__main__":
    main()
