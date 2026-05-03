"""Export Z-Image-Turbo transformer to ONNX for TensorRT compilation."""
import torch, os
import torch.distributed as dist
import torch._dynamo.utils as du

# Monkey patches
if not hasattr(dist, "device_mesh"):
    dist.device_mesh = type("device_mesh", (), {"DeviceMesh": type("FakeDM", (), {})})
if not hasattr(du, "NP_SUPPORTED_MODULES"):
    du.NP_SUPPORTED_MODULES = {}

from diffusers import ZImagePipeline

MODEL = "/models/z-image-turbo-fp8-diffusers"
ONNX_DIR = "/tmp/onnx-output"

os.makedirs(ONNX_DIR, exist_ok=True)

print("Loading model...", flush=True)
pipe = ZImagePipeline.from_pretrained(MODEL, torch_dtype=torch.float16)
pipe.to("cuda")

transformer = pipe.transformer
transformer.eval()

# Create dummy inputs matching model expectations
# ZImageTransformer2DModel forward signature:
# hidden_states, timestep, encoder_hidden_states, encoder_attention_mask, ...
B = 1
H = W = 64  # 512x512 latent
C = 16      # in_channels

hidden_states = torch.randn(B, C, H, W, dtype=torch.float16, device="cuda")
timestep = torch.tensor([500.0], dtype=torch.float16, device="cuda")
# Text encoder output: [B, seq_len, dim] where dim=3840
encoder_hidden_states = torch.randn(B, 77, 3840, dtype=torch.float16, device="cuda")
encoder_attention_mask = torch.ones(B, 77, dtype=torch.float16, device="cuda")

print(f"Input shapes: hs={hidden_states.shape}, t={timestep.shape}, enc={encoder_hidden_states.shape}", flush=True)

# Test forward pass first
print("Testing forward pass...", flush=True)
with torch.no_grad():
    output = transformer(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        return_dict=False,
    )
print(f"Output shape: {output[0].shape}, mean={output[0].mean().item():.4f}", flush=True)

# Export to ONNX
onnx_path = os.path.join(ONNX_DIR, "transformer_512.onnx")
print(f"Exporting to {onnx_path}...", flush=True)

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
        "encoder_attention_mask": {0: "batch", 1: "seq_len"},
        "output": {0: "batch", 2: "height", 3: "width"},
    },
    do_constant_folding=True,
    export_params=True,
)

import subprocess
result = subprocess.run(["ls", "-lh", onnx_path], capture_output=True, text=True)
print(result.stdout.strip())
print(f"ONNX export done: {onnx_path}")
