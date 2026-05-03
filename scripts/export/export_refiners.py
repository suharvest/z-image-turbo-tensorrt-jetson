#!/usr/bin/env python3
"""Export noise_refiner (2 layers) and context_refiner (2 layers) as BF16-ready ONNX.

noise_refiner: basic-mode global modulation, FP16 weights + FP32 AdaLN
context_refiner: no modulation, FP16 weights
"""

import os, sys, gc, types

os.environ.setdefault("https_proxy", "http://127.0.0.1:7890")
os.environ.setdefault("http_proxy", "http://127.0.0.1:7890")

import torch
import torch.nn as nn

# -- Monkey-patches for ONNX compat ----------------------------------------
import torch._library.custom_ops as _tco
_tco.custom_op = lambda name, mutates_args=None, device_types="": lambda fn: fn

_orig_pad_sequence = torch.nn.utils.rnn.pad_sequence
def _onnx_pad_sequence(sequences, batch_first=False, padding_value=0.0):
    if all(s.shape == sequences[0].shape for s in sequences):
        return torch.stack(sequences, dim=0 if batch_first else 1)
    return _orig_pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)
torch.nn.utils.rnn.pad_sequence = _onnx_pad_sequence

# -- Constants -------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "Tongyi-MAI/Z-Image-Turbo")
TRANSFORMER_SUBFOLDER = os.environ.get("TRANSFORMER_SUBFOLDER", "transformer")
RESOLUTION = int(os.environ.get("RESOLUTION", "512"))
LATENT_H = RESOLUTION // 8
LATENT_W = RESOLUTION // 8
IMG_H_TOKENS = LATENT_H // 2
IMG_W_TOKENS = LATENT_W // 2
IMG_TOKENS = IMG_H_TOKENS * IMG_W_TOKENS
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "onnx-512" if RESOLUTION == 512 else f"onnx-{RESOLUTION}",
)
FP16_CLAMP = 60000.0
B, DIM, HDIM, ADIM = 1, 3840, 128, 256
NR_SEQ = IMG_TOKENS   # noise_refiner token count
CR_SEQ = 128    # context_refiner token count

os.makedirs(OUTPUT_DIR, exist_ok=True)
FORCE_EXPORT = os.environ.get("FORCE_EXPORT", "0") == "1"

print(f"Resolution: {RESOLUTION} latent={LATENT_H}x{LATENT_W} image_tokens={IMG_TOKENS}", flush=True)
print(f"Model: {MODEL_PATH} subfolder={TRANSFORMER_SUBFOLDER or '<none>'}", flush=True)
print("PyTorch: " + torch.__version__ + ", CUDA: " + str(torch.cuda.is_available()), flush=True)
print("GPU: " + torch.cuda.get_device_name(0) + ", Memory free: " + str(torch.cuda.mem_get_info()[0]/1024**3) + " GB", flush=True)

from diffusers import ZImageTransformer2DModel
from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock
import onnx

# -- Load full model to CPU -------------------------------------------------
print("\nLoading transformer to CPU...", flush=True)
load_kwargs = {"torch_dtype": torch.float32}
if TRANSFORMER_SUBFOLDER:
    load_kwargs["subfolder"] = TRANSFORMER_SUBFOLDER
transformer = ZImageTransformer2DModel.from_pretrained(MODEL_PATH, **load_kwargs).to("cpu")
transformer.eval()
print("Model loaded.", flush=True)

# Derive config from actual layers
nr0 = transformer.noise_refiner[0]
cr0 = transformer.context_refiner[0]

n_heads = nr0.attention.heads
n_kv_heads = n_heads  # Z-Image-Turbo uses MHA (no GQA)
norm_eps = nr0.attention_norm1.eps

nr_config = {
    "layer_id": 0, "dim": nr0.dim, "n_heads": n_heads, "n_kv_heads": n_kv_heads,
    "norm_eps": norm_eps, "qk_norm": True,
    "modulation": True,
}
cr_config = {
    "layer_id": 0, "dim": cr0.dim, "n_heads": n_heads, "n_kv_heads": n_kv_heads,
    "norm_eps": norm_eps, "qk_norm": True,
    "modulation": False,
}
print("noise_refiner config:  dim=%d heads=%d modulation=True" % (nr_config['dim'], nr_config['n_heads']), flush=True)
print("context_refiner config: dim=%d heads=%d modulation=False" % (cr_config['dim'], cr_config['n_heads']), flush=True)

# -- Clamped FFN forward ----------------------------------------------------
def clamped_ffn_forward(self, x):
    x1 = torch.clamp(self.w1(x), -FP16_CLAMP, FP16_CLAMP)
    x3 = torch.clamp(self.w3(x), -FP16_CLAMP, FP16_CLAMP)
    gate = torch.clamp(self._forward_silu_gating(x1, x3), -FP16_CLAMP, FP16_CLAMP)
    return torch.clamp(self.w2(gate), -FP16_CLAMP, FP16_CLAMP)

# ===========================================================================
# NOISE_REFINER WRAPPER  (per-token modulation, noise_mask path)
# ===========================================================================
class NoiseRefinerWrapper(nn.Module):
    """Basic-mode global modulation with FP32 AdaLN, FP16 weights, clamped FFN."""

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

        # Attention block
        attn_out = self.layer.attention(
            self.layer.attention_norm1(x) * scale_msa,
            attention_mask=attn_mask, freqs_cis=freqs_cis,
        )
        x = x + gate_msa * self.layer.attention_norm2(attn_out)

        # FFN block
        x = x + gate_mlp * self.layer.ffn_norm2(
            self.layer.feed_forward(self.layer.ffn_norm1(x) * scale_mlp)
        )
        return x

# ===========================================================================
# CONTEXT_REFINER WRAPPER  (no modulation, modulation=False)
# ===========================================================================
class ContextRefinerWrapper(nn.Module):
    """No modulation, FP16 weights, clamped FFN."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, attn_mask, freqs_cis):
        # Attention (no modulation)
        attn_out = self.layer.attention(
            self.layer.attention_norm1(x),
            attention_mask=attn_mask, freqs_cis=freqs_cis,
        )
        x = x + self.layer.attention_norm2(attn_out)

        # FFN (no modulation)
        x = x + self.layer.ffn_norm2(
            self.layer.feed_forward(self.layer.ffn_norm1(x))
        )
        return x

# ===========================================================================
# EXPORT HELPERS
# ===========================================================================

def export_noise_refiner(idx, source_layer, config):
    name = "noise_refiner_%02d" % idx
    onnx_path = os.path.join(OUTPUT_DIR, name + ".onnx")

    if os.path.exists(onnx_path) and not FORCE_EXPORT:
        print("[%s] SKIP (already exists)" % name, flush=True)
        return True

    sep = "=" * 60
    print("\n" + sep, flush=True)
    print("[%s] Exporting noise_refiner[%d]..." % (name, idx), flush=True)

    try:
        # Extract state dict
        sd = {k: v.clone() for k, v in source_layer.state_dict().items()}

        # Create standalone layer with FP16 weights
        layer = ZImageTransformerBlock(**config)
        layer.load_state_dict({k: v.to(torch.float16) for k, v in sd.items()})
        del sd; gc.collect()

        # Monkey-patch FFN
        layer.feed_forward.forward = types.MethodType(clamped_ffn_forward, layer.feed_forward)

        # Move to GPU: FP16 weights, FP32 AdaLN
        layer = layer.to("cuda").to(torch.float16)
        for p in layer.adaLN_modulation.parameters():
            p.data = p.data.to(torch.float32)
        layer.eval()

        wrapper = NoiseRefinerWrapper(layer)
        wrapper.eval()

        # Test forward pass
        x = torch.randn(B, NR_SEQ, DIM, dtype=torch.float16, device="cuda") * 0.1
        mask = torch.ones(B, NR_SEQ, dtype=torch.bool, device="cuda")
        freqs = torch.randn(B, NR_SEQ, HDIM, dtype=torch.float16, device="cuda") * 0.1
        adaln = torch.randn(B, ADIM, dtype=torch.float16, device="cuda") * 0.1

        with torch.no_grad():
            out = wrapper(x, mask, freqs, adaln)

        ok = not (torch.isnan(out).any().item() or torch.isinf(out).any().item())
        print("  Test: %s  mean=%.6f  std=%.6f" % (
            "OK" if ok else "FAIL", out.float().mean().item(), out.float().std().item()), flush=True)

        if not ok:
            raise RuntimeError("NaN/Inf in forward pass")

        # Export ONNX
        torch.onnx.export(
            wrapper, (x, mask, freqs, adaln),
            onnx_path,
            input_names=["x", "attn_mask", "freqs_cis", "adaln_input"],
            output_names=["output"],
            opset_version=17,
            dynamic_axes=None,
            do_constant_folding=True,
            export_params=True,
        )

        # Verify
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        fp16_w = sum(1 for i in model.graph.initializer if i.data_type == 10)
        fp32_w = sum(1 for i in model.graph.initializer if i.data_type == 1)
        size_mb = os.path.getsize(onnx_path) / 1024**2
        has_clip = "Clip" in set(n.op_type for n in model.graph.node)
        print("  ONNX: %.1f MB | %d FP16 + %d FP32 | Clip:%s | OK" % (size_mb, fp16_w, fp32_w, has_clip), flush=True)

        del wrapper, layer, out
        gc.collect(); torch.cuda.empty_cache()
        return True

    except Exception as e:
        print("  ERROR: " + str(e), flush=True)
        import traceback; traceback.print_exc()
        gc.collect(); torch.cuda.empty_cache()
        return False


def export_context_refiner(idx, source_layer, config):
    name = "context_refiner_%02d" % idx
    onnx_path = os.path.join(OUTPUT_DIR, name + ".onnx")

    if os.path.exists(onnx_path) and not FORCE_EXPORT:
        print("[%s] SKIP (already exists)" % name, flush=True)
        return True

    sep = "=" * 60
    print("\n" + sep, flush=True)
    print("[%s] Exporting context_refiner[%d]..." % (name, idx), flush=True)

    try:
        # Extract state dict
        sd = {k: v.clone() for k, v in source_layer.state_dict().items()}

        # Create standalone layer with FP16 weights
        layer = ZImageTransformerBlock(**config)
        layer.load_state_dict({k: v.to(torch.float16) for k, v in sd.items()})
        del sd; gc.collect()

        # Monkey-patch FFN
        layer.feed_forward.forward = types.MethodType(clamped_ffn_forward, layer.feed_forward)

        # Move to GPU: all FP16 (no adaLN for modulation=False)
        layer = layer.to("cuda").to(torch.float16)
        layer.eval()

        wrapper = ContextRefinerWrapper(layer)
        wrapper.eval()

        # Test forward pass
        x = torch.randn(B, CR_SEQ, DIM, dtype=torch.float16, device="cuda") * 0.1
        mask = torch.ones(B, CR_SEQ, dtype=torch.bool, device="cuda")
        freqs = torch.randn(B, CR_SEQ, HDIM, dtype=torch.float16, device="cuda") * 0.1

        with torch.no_grad():
            out = wrapper(x, mask, freqs)

        ok = not (torch.isnan(out).any().item() or torch.isinf(out).any().item())
        print("  Test: %s  mean=%.6f  std=%.6f" % (
            "OK" if ok else "FAIL", out.float().mean().item(), out.float().std().item()), flush=True)

        if not ok:
            raise RuntimeError("NaN/Inf in forward pass")

        # Export ONNX
        torch.onnx.export(
            wrapper, (x, mask, freqs),
            onnx_path,
            input_names=["x", "attn_mask", "freqs_cis"],
            output_names=["output"],
            opset_version=17,
            dynamic_axes=None,
            do_constant_folding=True,
            export_params=True,
        )

        # Verify
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        fp16_w = sum(1 for i in model.graph.initializer if i.data_type == 10)
        fp32_w = sum(1 for i in model.graph.initializer if i.data_type == 1)
        size_mb = os.path.getsize(onnx_path) / 1024**2
        has_clip = "Clip" in set(n.op_type for n in model.graph.node)
        print("  ONNX: %.1f MB | %d FP16 + %d FP32 | Clip:%s | OK" % (size_mb, fp16_w, fp32_w, has_clip), flush=True)

        del wrapper, layer, out
        gc.collect(); torch.cuda.empty_cache()
        return True

    except Exception as e:
        print("  ERROR: " + str(e), flush=True)
        import traceback; traceback.print_exc()
        gc.collect(); torch.cuda.empty_cache()
        return False


# ===========================================================================
# MAIN
# ===========================================================================
sep = "=" * 60
print("\n" + sep, flush=True)
print("EXPORTING 4 REFINER BLOCKS", flush=True)
print(sep, flush=True)

results = []
results.append(("noise_refiner_00",  export_noise_refiner(0, transformer.noise_refiner[0], nr_config)))
results.append(("noise_refiner_01",  export_noise_refiner(1, transformer.noise_refiner[1], nr_config)))
results.append(("context_refiner_00", export_context_refiner(0, transformer.context_refiner[0], cr_config)))
results.append(("context_refiner_01", export_context_refiner(1, transformer.context_refiner[1], cr_config)))

print("\n" + sep, flush=True)
print("EXPORT SUMMARY", flush=True)
print(sep, flush=True)
passed = 0
for name, ok in results:
    status = "PASSED" if ok else "FAILED"
    if ok: passed += 1
    print("  %s: %s" % (name, status), flush=True)
print("\nPassed: %d/%d" % (passed, len(results)), flush=True)
