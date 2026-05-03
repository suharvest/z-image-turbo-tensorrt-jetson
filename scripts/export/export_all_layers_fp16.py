#!/usr/bin/env python3
"""Batch export ALL 30 Z-Image-Turbo transformer layers + pre/post components as FP16 ONNX.

Each layer: x[1,1152,3840] + attn_mask[1,1152] + freqs_cis[1,1152,128] + adaln_input[1,256] -> output[1,1152,3840]
Pre/post: t_embedder, prompt_preprocessor, latent_preprocessor, final_projection
"""

import os, sys, gc, types, math

os.environ.setdefault("https_proxy", "http://127.0.0.1:7890")
os.environ.setdefault("http_proxy", "http://127.0.0.1:7890")

import torch
import torch.nn as nn
import torch.nn.functional as F

# -- Monkey-patches for ONNX compat ------------------------------------------
import torch._library.custom_ops as _tco
_tco.custom_op = lambda name, mutates_args=None, device_types="": lambda fn: fn

_orig_pad_sequence = torch.nn.utils.rnn.pad_sequence
def _onnx_pad_sequence(sequences, batch_first=False, padding_value=0.0):
    if all(s.shape == sequences[0].shape for s in sequences):
        return torch.stack(sequences, dim=0 if batch_first else 1)
    return _orig_pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)
torch.nn.utils.rnn.pad_sequence = _onnx_pad_sequence

# -- Constants ---------------------------------------------------------------
MODEL_PATH = "/home/harve/models/huggingface/models--Tongyi-MAI--Z-Image-Turbo/snapshots/f332072aa78be7aecdf3ee76d5c247082da564a6/transformer"
RESOLUTION = int(os.environ.get("RESOLUTION", "512"))
LATENT_H = RESOLUTION // 8
LATENT_W = RESOLUTION // 8
IMG_H_TOKENS = LATENT_H // 2
IMG_W_TOKENS = LATENT_W // 2
IMG_TOKENS = IMG_H_TOKENS * IMG_W_TOKENS
TEXT_TOKENS = int(os.environ.get("TEXT_TOKENS", "128"))
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "/home/harve/trt-work/onnx-layers" if RESOLUTION == 512 else f"/home/harve/trt-work/onnx-{RESOLUTION}",
)
FP16_CLAMP = 60000.0
B, SEQ, DIM, HDIM, ADIM = 1, IMG_TOKENS + TEXT_TOKENS, 3840, 128, 256
N_LAYERS = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Resolution: {RESOLUTION} latent={LATENT_H}x{LATENT_W} image_tokens={IMG_TOKENS} seq={SEQ}", flush=True)
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
print(f"GPU: {torch.cuda.get_device_name(0)}, Memory free: {torch.cuda.mem_get_info()[0]/1024**3:.1f} GB", flush=True)

from diffusers import ZImageTransformer2DModel
from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock

import onnx

# -- Load full model (CPU, to save VRAM) ------------------------------------
print("\nLoading transformer to CPU...", flush=True)
transformer = ZImageTransformer2DModel.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float32,
).to("cpu")
transformer.eval()
print(f"Model loaded. {len(transformer.layers)} layers.", flush=True)

# -- FFN clamp patch (applied per-layer at runtime) -------------------------
def make_clamped_ffn_forward(self, x):
    x1 = torch.clamp(self.w1(x), -FP16_CLAMP, FP16_CLAMP)
    x3 = torch.clamp(self.w3(x), -FP16_CLAMP, FP16_CLAMP)
    gate = torch.clamp(self._forward_silu_gating(x1, x3), -FP16_CLAMP, FP16_CLAMP)
    out = torch.clamp(self.w2(gate), -FP16_CLAMP, FP16_CLAMP)
    return out

# -- Layer wrapper -----------------------------------------------------------
class SingleLayerWrapper(nn.Module):
    """Wraps ZImageTransformerBlock for basic-mode ONNX export."""
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
            attention_mask=attn_mask, freqs_cis=freqs_cis,
        )
        x = x + gate_msa * self.layer.attention_norm2(attn_out)
        x = x + gate_mlp * self.layer.ffn_norm2(
            self.layer.feed_forward(self.layer.ffn_norm1(x) * scale_mlp)
        )
        return x


# ===========================================================================
# PART 1: Export all 30 transformer layers
# ===========================================================================

layer_config = {
    "dim": DIM, "n_heads": 30, "n_kv_heads": 30,
    "norm_eps": 1e-5, "qk_norm": True, "modulation": True,
}

errors = []
for layer_id in range(N_LAYERS):
    print(f"\n{'='*60}")
    print(f"[{layer_id:02d}/29] Exporting layer {layer_id}...", flush=True)

    try:
        # Extract state dict from CPU model
        layer_sd = {k: v.clone() for k, v in transformer.layers[layer_id].state_dict().items()}

        # Create fresh layer on GPU with FP16 weights
        layer_config["layer_id"] = layer_id
        layer = ZImageTransformerBlock(**layer_config)
        layer.load_state_dict({k: v.to(torch.float16) for k, v in layer_sd.items()})
        del layer_sd
        gc.collect()

        # Monkey-patch FFN
        layer.feed_forward.forward = types.MethodType(make_clamped_ffn_forward, layer.feed_forward)

        layer = layer.to("cuda").to(torch.float16)
        for p in layer.adaLN_modulation.parameters():
            p.data = p.data.to(torch.float32)
        layer.eval()

        # Wrap
        wrapper = SingleLayerWrapper(layer)
        wrapper.eval()

        # Test forward pass
        x = torch.randn(B, SEQ, DIM, dtype=torch.float16, device="cuda")
        freqs = torch.randn(B, SEQ, HDIM, dtype=torch.float16, device="cuda")
        adaln = torch.randn(B, ADIM, dtype=torch.float16, device="cuda")
        attn_mask = torch.ones(B, SEQ, dtype=torch.bool, device="cuda")

        with torch.no_grad():
            out = wrapper(x, attn_mask, freqs, adaln)

        ok = not (torch.isnan(out).any().item() or torch.isinf(out).any().item())
        print(f"  Test: {'OK' if ok else 'FAIL'}  mean={out.float().mean().item():.6f}  std={out.float().std().item():.6f}", flush=True)

        # Export ONNX
        onnx_path = os.path.join(OUTPUT_DIR, f"layer_{layer_id:02d}_fp16.onnx")
        torch.onnx.export(
            wrapper, (x, attn_mask, freqs, adaln),
            onnx_path,
            input_names=["x", "attn_mask", "freqs_cis", "adaln_input"],
            output_names=["output"],
            opset_version=17,
            dynamic_axes=None,
            do_constant_folding=True,
            export_params=True,
        )

        # Verify ONNX
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        fp16_w = sum(1 for i in model.graph.initializer if i.data_type == 10)
        fp32_w = sum(1 for i in model.graph.initializer if i.data_type == 1)
        clip_nodes = sum(1 for n in model.graph.node if n.op_type == "Clip")
        size_mb = os.path.getsize(onnx_path) / 1024**2
        print(f"  ONNX: {size_mb:.1f} MB  |  {fp16_w} FP16 + {fp32_w} FP32 weights  |  {clip_nodes} Clip ops  |  OK", flush=True)

        del wrapper, layer, out, x, freqs, adaln, attn_mask
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ERROR exporting layer {layer_id}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        errors.append((layer_id, str(e)))
        gc.collect()
        torch.cuda.empty_cache()
        continue

# ===========================================================================
# PART 2: Pre/Post processing components
# ===========================================================================

print(f"\n{'='*60}")
print("Exporting pre/post processing components...", flush=True)

# -- 2a: t_embedder (timestep MLP) ------------------------------------------
# Input: timestep [1] (scalar)  ->  Output: adaln_input [1, 256]
print("\n--- t_embedder ---", flush=True)
try:
    t_embedder = transformer.t_embedder.to("cuda").to(torch.float16)
    t_embedder.eval()

    class TEmbedderWrapper(nn.Module):
        def __init__(self, embedder):
            super().__init__()
            self.freq_dim = embedder.frequency_embedding_size
            self.mlp = embedder.mlp

        def forward(self, t):
            # Manual timestep_embedding (avoid autocast for ONNX trace)
            half = self.freq_dim // 2
            freqs = torch.exp(
                -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if self.freq_dim % 2:
                t_freq = torch.cat([t_freq, torch.zeros_like(t_freq[:, :1])], dim=-1)
            t_freq = t_freq.to(torch.float16)
            return self.mlp(t_freq)

    tw = TEmbedderWrapper(t_embedder)
    tw.eval()

    t_in = torch.tensor([500.0], dtype=torch.float32, device="cuda")
    with torch.no_grad():
        t_out = tw(t_in)
    print(f"  Test: shape={t_out.shape}  mean={t_out.float().mean().item():.6f}", flush=True)

    onnx_path = os.path.join(OUTPUT_DIR, "t_embedder_fp16.onnx")
    torch.onnx.export(
        tw, (t_in,), onnx_path,
        input_names=["timestep"], output_names=["adaln_input"],
        opset_version=17, dynamic_axes=None,
        do_constant_folding=True, export_params=True,
    )
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f"  ONNX: {os.path.getsize(onnx_path)/1024**2:.1f} MB  OK", flush=True)

    del tw, t_embedder, t_in, t_out
    gc.collect()
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    errors.append(("t_embedder", str(e)))

# -- 2b: prompt_preprocessor (cap_embedder: text projection) -----------------
# Input: prompt_embeds [1, 128, 2560]  ->  Output: processed_prompt [1, 128, 3840]
print("\n--- prompt_preprocessor ---", flush=True)
try:
    cap_embedder = transformer.cap_embedder.to("cuda").to(torch.float16)
    cap_embedder.eval()

    prompt_in = torch.randn(1, 128, 2560, dtype=torch.float16, device="cuda")
    with torch.no_grad():
        prompt_out = cap_embedder(prompt_in)
    print(f"  Test: shape={prompt_out.shape}  mean={prompt_out.float().mean().item():.6f}", flush=True)

    onnx_path = os.path.join(OUTPUT_DIR, "prompt_preprocessor_fp16.onnx")
    torch.onnx.export(
        cap_embedder, (prompt_in,), onnx_path,
        input_names=["prompt_embeds"], output_names=["processed_prompt"],
        opset_version=17, dynamic_axes=None,
        do_constant_folding=True, export_params=True,
    )
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f"  ONNX: {os.path.getsize(onnx_path)/1024**2:.1f} MB  OK", flush=True)

    del cap_embedder, prompt_in, prompt_out
    gc.collect()
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    errors.append(("prompt_preprocessor", str(e)))

# -- 2c: latent_preprocessor (patchify + x_embedder) -------------------------
# Input: latent [1, 16, 64, 64]  ->  Output: image_tokens [1, 1024, 3840]
print("\n--- latent_preprocessor ---", flush=True)
try:
    # Get the x_embedder for patch_size=2, f_patch_size=1
    x_embedder_key = "2-1"
    x_embedder = transformer.all_x_embedder[x_embedder_key]
    x_embedder_weight = x_embedder.weight.data.clone()
    x_embedder_bias = x_embedder.bias.data.clone()

    class LatentPreprocessor(nn.Module):
        def __init__(self, weight, bias, in_channels=16, patch_size=2, dim=3840):
            super().__init__()
            self.linear = nn.Linear(patch_size * patch_size * in_channels, dim, bias=True)
            self.linear.weight.data = weight
            self.linear.bias.data = bias
            self.patch_size = patch_size
            self.in_channels = in_channels

        def forward(self, latent):
            # latent: [B, C, H, W]
            B, C, H, W = latent.shape
            pH = self.patch_size
            pW = self.patch_size
            H_tokens = H // pH
            W_tokens = W // pW
            # Patchify: [B, C, H, W] -> [B, C, H/ph, ph, W/pw, pw]
            # -> [B, H/ph, W/pw, ph, pw, C] -> [B, H/ph*W/pw, ph*pw*C]
            x = latent.view(B, C, H_tokens, pH, W_tokens, pW)
            x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, H_tokens * W_tokens, pH * pW * C)
            return self.linear(x.to(torch.float16))

    lp = LatentPreprocessor(x_embedder_weight, x_embedder_bias)
    lp = lp.to("cuda").to(torch.float16)
    lp.eval()

    latent_in = torch.randn(1, 16, LATENT_H, LATENT_W, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        latent_out = lp(latent_in)
    print(f"  Test: shape={latent_out.shape}  mean={latent_out.float().mean().item():.6f}", flush=True)

    onnx_path = os.path.join(OUTPUT_DIR, "latent_preprocessor_fp16.onnx")
    torch.onnx.export(
        lp, (latent_in,), onnx_path,
        input_names=["latent"], output_names=["image_tokens"],
        opset_version=17, dynamic_axes=None,
        do_constant_folding=True, export_params=True,
    )
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f"  ONNX: {os.path.getsize(onnx_path)/1024**2:.1f} MB  OK", flush=True)

    del lp, latent_in, latent_out, x_embedder_weight, x_embedder_bias
    gc.collect()
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    errors.append(("latent_preprocessor", str(e)))

# -- 2d: final_projection (FinalLayer + unpatchify) --------------------------
# Input: hidden [1, 1152, 3840] + adaln_input [1, 256]
# Output: noise_pred [1, 16, 64, 64]
print("\n--- final_projection ---", flush=True)
try:
    final_key = "2-1"
    final_layer = transformer.all_final_layer[final_key]
    final_sd = {k: v.clone() for k, v in final_layer.state_dict().items()}

    class FinalProjection(nn.Module):
        def __init__(
            self, state_dict, dim=3840, out_channels=64, adaln_dim=256,
            img_tokens=IMG_TOKENS, h_tokens=IMG_H_TOKENS, w_tokens=IMG_W_TOKENS,
        ):
            super().__init__()
            self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.linear = nn.Linear(dim, out_channels, bias=True)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(adaln_dim, dim, bias=True),
            )
            self.load_state_dict(state_dict)
            self.img_tokens = img_tokens
            self.h_tokens = h_tokens
            self.w_tokens = w_tokens
            self.out_channels = 16  # in_channels
            self.patch_size = 2

        def forward(self, hidden, adaln_input):
            # FinalLayer logic
            scale = 1.0 + self.adaLN_modulation(adaln_input)
            scale = scale.unsqueeze(1)  # [B, 1, dim]
            x = self.norm_final(hidden) * scale
            x = self.linear(x)  # [B, seq, out_channels]

            # Extract image tokens (first img_tokens)
            x = x[:, :self.img_tokens, :]

            # Unpatchify: [B, image_tokens, 64] -> [B, 16, latent_h, latent_w]
            B = x.shape[0]
            pH = pW = self.patch_size
            x = x.view(B, self.h_tokens, self.w_tokens, pH, pW, self.out_channels)
            x = x.permute(0, 5, 1, 3, 2, 4).reshape(
                B, self.out_channels, self.h_tokens * pH, self.w_tokens * pW
            )
            return x

    fp = FinalProjection(final_sd)
    fp = fp.to("cuda").to(torch.float16)
    fp.eval()
    del final_sd

    hidden_in = torch.randn(1, SEQ, 3840, dtype=torch.float16, device="cuda")
    adaln_in = torch.randn(1, 256, dtype=torch.float16, device="cuda")
    with torch.no_grad():
        noise_out = fp(hidden_in, adaln_in)
    print(f"  Test: shape={noise_out.shape}  mean={noise_out.float().mean().item():.6f}", flush=True)

    onnx_path = os.path.join(OUTPUT_DIR, "final_projection_fp16.onnx")
    torch.onnx.export(
        fp, (hidden_in, adaln_in), onnx_path,
        input_names=["hidden", "adaln_input"], output_names=["noise_pred"],
        opset_version=17, dynamic_axes=None,
        do_constant_folding=True, export_params=True,
    )
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f"  ONNX: {os.path.getsize(onnx_path)/1024**2:.1f} MB  OK", flush=True)

    del fp, hidden_in, adaln_in, noise_out
    gc.collect()
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    errors.append(("final_projection", str(e)))

# -- Cleanup ----------------------------------------------------------------
del transformer
gc.collect()

# -- Summary ----------------------------------------------------------------
print(f"\n{'='*60}")
print("EXPORT SUMMARY")
print(f"{'='*60}")
onnx_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".onnx")])
for f in onnx_files:
    size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024**2
    print(f"  {f:45s}  {size_mb:8.1f} MB")

if errors:
    print(f"\nERRORS ({len(errors)}):")
    for name, err in errors:
        print(f"  - {name}: {err[:120]}")

total_mb = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in onnx_files) / 1024
print(f"\nTotal: {len(onnx_files)} files, {total_mb:.1f} MB")
print("DONE", flush=True)
