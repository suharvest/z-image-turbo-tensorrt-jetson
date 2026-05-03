"""ONNX export for Z-Image-Turbo transformer on wsl2-local.
Downloads model from HuggingFace, exports transformer to ONNX.
"""
import os, sys, torch
import torch.onnx

# Proxy for HuggingFace
os.environ.setdefault("https_proxy", "http://127.0.0.1:7890")
os.environ.setdefault("http_proxy", "http://127.0.0.1:7890")

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
OUTPUT_DIR = "/home/harvest/trt-work/onnx-output"
CACHE_DIR = "/home/harvest/models/huggingface"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Step 1: Download transformer only from HuggingFace
print("=== Downloading transformer from HuggingFace ===", flush=True)
from huggingface_hub import snapshot_download

local_path = snapshot_download(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    allow_patterns=["transformer/*", "model_index.json"],
    local_files_only=False,
)
print(f"Downloaded to: {local_path}", flush=True)

# Step 2: Load transformer directly (skip full pipeline to save RAM)
print("=== Loading transformer ===", flush=True)
from diffusers import ZImageTransformer2DModel

transformer = ZImageTransformer2DModel.from_pretrained(
    local_path,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
transformer.eval()
print(f"Transformer loaded. Params: {sum(p.numel() for p in transformer.parameters())/1e9:.1f}B", flush=True)

# Step 3: Create dummy inputs
print("=== Creating dummy inputs ===", flush=True)
B = 1
H = W = 64   # 512x512 → latent 64x64
C = 16        # in_channels

hidden_states = torch.randn(B, C, H, W, dtype=torch.bfloat16)
timestep = torch.tensor([500.0], dtype=torch.bfloat16)
encoder_hidden_states = torch.randn(B, 77, 3840, dtype=torch.bfloat16)
encoder_attention_mask = torch.ones(B, 77, dtype=torch.bfloat16)

print(f"hs={hidden_states.shape}, t={timestep.shape}, enc={encoder_hidden_states.shape}", flush=True)

# Step 4: Test forward pass (CPU - no GPU needed for ONNX trace)
print("=== Testing forward pass (CPU) ===", flush=True)
with torch.no_grad():
    output = transformer(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        return_dict=False,
    )
print(f"Output shape: {output[0].shape}, mean={output[0].mean().item():.4f}", flush=True)

# Step 5: Export to ONNX
onnx_path = os.path.join(OUTPUT_DIR, "transformer_512.onnx")
print(f"=== Exporting ONNX to {onnx_path} ===", flush=True)

# For large models, use dynamo export which handles more ops
torch.onnx.export(
    transformer,
    (hidden_states, timestep, encoder_hidden_states, encoder_attention_mask),
    onnx_path,
    input_names=["hidden_states", "timestep", "encoder_hidden_states", "attention_mask"],
    output_names=["output"],
    opset_version=18,
    dynamic_axes={
        "hidden_states": {0: "batch", 2: "height", 3: "width"},
        "timestep": {0: "batch"},
        "encoder_hidden_states": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "output": {0: "batch", 2: "height", 3: "width"},
    },
    do_constant_folding=True,
    export_params=True,
)

import subprocess
result = subprocess.run(["ls", "-lh", onnx_path], capture_output=True, text=True)
print(result.stdout.strip(), flush=True)
print(f"=== ONNX export complete: {onnx_path} ===", flush=True)
