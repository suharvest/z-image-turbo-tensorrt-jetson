#!/usr/bin/env python3
"""Export grouped Z-Image-Turbo transformer layers for TensorRT.

Prototype target:
  layers_00_04_fp16.onnx

Inputs:
  x [1,1152,3840] FP16
  attn_mask [1,1152] BOOL
  freqs_cis [1,1152,128] FP16
  adaln_input [1,256] FP16

Output:
  output [1,1152,3840] FP16
"""

import gc
import math
import os
import sys
import types

os.environ.setdefault("https_proxy", "http://127.0.0.1:7890")
os.environ.setdefault("http_proxy", "http://127.0.0.1:7890")

import torch
import torch.nn as nn

import torch._library.custom_ops as _tco

_tco.custom_op = lambda name, mutates_args=None, device_types="": lambda fn: fn

_orig_pad_sequence = torch.nn.utils.rnn.pad_sequence


def _onnx_pad_sequence(sequences, batch_first=False, padding_value=0.0):
    if all(s.shape == sequences[0].shape for s in sequences):
        return torch.stack(sequences, dim=0 if batch_first else 1)
    return _orig_pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)


torch.nn.utils.rnn.pad_sequence = _onnx_pad_sequence

MODEL_PATH = os.environ.get("MODEL_PATH", "Tongyi-MAI/Z-Image-Turbo")
TRANSFORMER_SUBFOLDER = os.environ.get("TRANSFORMER_SUBFOLDER", "transformer")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "onnx-group-layers")
START_LAYER = int(os.environ.get("START_LAYER", "0"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "5"))
FP16_CLAMP = 60000.0
B, SEQ, DIM, HDIM, ADIM = 1, 1152, 3840, 128, 256

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
print(f"Model: {MODEL_PATH} subfolder={TRANSFORMER_SUBFOLDER or '<none>'}", flush=True)
print(f"GPU: {torch.cuda.get_device_name(0)}, Memory free: {torch.cuda.mem_get_info()[0]/1024**3:.1f} GB", flush=True)

from diffusers import ZImageTransformer2DModel
from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock
import onnx


def clamped_ffn_forward(self, x):
    x1 = torch.clamp(self.w1(x), -FP16_CLAMP, FP16_CLAMP)
    x3 = torch.clamp(self.w3(x), -FP16_CLAMP, FP16_CLAMP)
    gate = torch.clamp(self._forward_silu_gating(x1, x3), -FP16_CLAMP, FP16_CLAMP)
    return torch.clamp(self.w2(gate), -FP16_CLAMP, FP16_CLAMP)


class SingleLayerWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, attn_mask, freqs_cis, adaln_input):
        mod = self.layer.adaLN_modulation(adaln_input.to(torch.float32))
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
        scale_msa, gate_msa = scale_msa.to(torch.float16), gate_msa.to(torch.float16)
        scale_mlp, gate_mlp = scale_mlp.to(torch.float16), gate_mlp.to(torch.float16)

        attn_out = self.layer.attention(
            self.layer.attention_norm1(x) * scale_msa,
            attention_mask=attn_mask,
            freqs_cis=freqs_cis,
        )
        x = x + gate_msa * self.layer.attention_norm2(attn_out)
        x = x + gate_mlp * self.layer.ffn_norm2(
            self.layer.feed_forward(self.layer.ffn_norm1(x) * scale_mlp)
        )
        return x


class GroupLayerWrapper(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList([SingleLayerWrapper(layer) for layer in layers])

    def forward(self, x, attn_mask, freqs_cis, adaln_input):
        for layer in self.layers:
            x = layer(x, attn_mask, freqs_cis, adaln_input)
        return x


def make_layer(source_layer, layer_id):
    config = {
        "layer_id": layer_id,
        "dim": DIM,
        "n_heads": 30,
        "n_kv_heads": 30,
        "norm_eps": 1e-5,
        "qk_norm": True,
        "modulation": True,
    }
    sd = {k: v.clone() for k, v in source_layer.state_dict().items()}
    layer = ZImageTransformerBlock(**config)
    layer.load_state_dict({k: v.to(torch.float16) for k, v in sd.items()})
    layer.feed_forward.forward = types.MethodType(clamped_ffn_forward, layer.feed_forward)
    layer = layer.to("cuda").to(torch.float16)
    for p in layer.adaLN_modulation.parameters():
        p.data = p.data.to(torch.float32)
    layer.eval()
    return layer


def main():
    end_layer = START_LAYER + GROUP_SIZE - 1
    name = f"layers_{START_LAYER:02d}_{end_layer:02d}_fp16"
    onnx_path = os.path.join(OUTPUT_DIR, name + ".onnx")

    print(f"Loading transformer CPU: {MODEL_PATH}", flush=True)
    load_kwargs = {"torch_dtype": torch.float32}
    if TRANSFORMER_SUBFOLDER:
        load_kwargs["subfolder"] = TRANSFORMER_SUBFOLDER
    transformer = ZImageTransformer2DModel.from_pretrained(MODEL_PATH, **load_kwargs).to("cpu")
    transformer.eval()

    layers = []
    for layer_id in range(START_LAYER, START_LAYER + GROUP_SIZE):
        print(f"Preparing layer {layer_id:02d}", flush=True)
        layers.append(make_layer(transformer.layers[layer_id], layer_id))
        gc.collect()
        torch.cuda.empty_cache()

    wrapper = GroupLayerWrapper(layers).eval()

    x = torch.randn(B, SEQ, DIM, dtype=torch.float16, device="cuda")
    freqs = torch.randn(B, SEQ, HDIM, dtype=torch.float16, device="cuda")
    adaln = torch.randn(B, ADIM, dtype=torch.float16, device="cuda")
    attn_mask = torch.ones(B, SEQ, dtype=torch.bool, device="cuda")

    with torch.no_grad():
        out = wrapper(x, attn_mask, freqs, adaln)
    if torch.isnan(out).any().item() or torch.isinf(out).any().item():
        raise RuntimeError("NaN/Inf in grouped forward")
    print(f"Test OK: mean={out.float().mean().item():.6f} std={out.float().std().item():.6f}", flush=True)

    print(f"Exporting {onnx_path}", flush=True)
    torch.onnx.export(
        wrapper,
        (x, attn_mask, freqs, adaln),
        onnx_path,
        input_names=["x", "attn_mask", "freqs_cis", "adaln_input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes=None,
        do_constant_folding=True,
        export_params=True,
    )

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    fp16_w = sum(1 for i in model.graph.initializer if i.data_type == 10)
    fp32_w = sum(1 for i in model.graph.initializer if i.data_type == 1)
    clip_nodes = sum(1 for n in model.graph.node if n.op_type == "Clip")
    print(
        f"ONNX OK: {os.path.getsize(onnx_path)/1024**2:.1f} MB | "
        f"{fp16_w} FP16 + {fp32_w} FP32 | {clip_nodes} Clip ops",
        flush=True,
    )


if __name__ == "__main__":
    main()
