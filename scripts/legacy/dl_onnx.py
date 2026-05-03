"""Download one AXERA split ONNX for TRT testing."""
import os, sys

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

print("Installing huggingface_hub...", flush=True)
os.system("pip3 install -q huggingface_hub 2>/dev/null")

from huggingface_hub import hf_hub_download

# Download smallest split (cfg_00, 2.11MB - t_embedder)
FILE = "transformer_onnx/cfg_00_timestep_to_model_t_embedder_mlp_mlp_2_Gemm_output_0_config.onnx"
DEST = "/home/harvest/models/axera-onnx"

print(f"Downloading {FILE}...", flush=True)
path = hf_hub_download(
    "AXERA-TECH/Z-Image-Turbo",
    FILE,
    local_dir=DEST,
    local_dir_use_symlinks=False,
)
print(f"OK: {path} ({os.path.getsize(path)/1e6:.1f}MB)", flush=True)

# Also download one layer split (cfg_03, 724MB)
LAYER_FILE = "transformer_onnx/cfg_03_model_Slice_1_output_0_to_model_layers_0_Add_4_output_0_config.onnx"
print(f"Downloading {LAYER_FILE}...", flush=True)
path2 = hf_hub_download(
    "AXERA-TECH/Z-Image-Turbo",
    LAYER_FILE,
    local_dir=DEST,
    local_dir_use_symlinks=False,
)
print(f"OK: {path2} ({os.path.getsize(path2)/1e6:.1f}MB)", flush=True)
print("DONE", flush=True)
