#!/bin/bash
# ONNX export setup script for wsl2-local
# Downloads FP8 model, converts to BF16 diffusers, exports ONNX
set -euo pipefail

WORK_DIR="/home/harvest/trt-work"
MODEL_DIR="/home/harvest/models"
FP8_DIR="$MODEL_DIR/z-image-turbo-fp8"
BF16_DIR="$MODEL_DIR/z-image-turbo-fp8-diffusers"
ONNX_DIR="$WORK_DIR/onnx-output"

mkdir -p "$WORK_DIR" "$MODEL_DIR" "$ONNX_DIR"

# Setup proxy
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890

echo "=== Step 1: Download FP8 model from HuggingFace ==="
if [ -d "$FP8_DIR" ] && [ -f "$FP8_DIR/diffusion_pytorch_model.safetensors" ]; then
    echo "FP8 model already exists, skipping download"
else
    python3 -c "
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import snapshot_download
snapshot_download('drbaph/Z-Image-Turbo-FP8', local_dir='$FP8_DIR', local_dir_use_symlinks=False)
print('Download complete')
"
fi

echo "=== Step 2: Check model files ==="
ls -lh "$FP8_DIR/" 2>/dev/null | head -20
du -sh "$FP8_DIR/" 2>/dev/null

echo "=== Step 3: Convert FP8 to BF16 diffusers format ==="
if [ -d "$BF16_DIR" ] && [ -f "$BF16_DIR/transformer/diffusion_pytorch_model.safetensors" ]; then
    echo "BF16 model already exists, skipping conversion"
else
    cd "$WORK_DIR"
    python3 -u convert_fp8_stream.py
fi

echo "=== Step 4: Export transformer to ONNX ==="
cd "$WORK_DIR"
python3 -u export_onnx.py

echo "=== Step 5: Verify ONNX output ==="
ls -lh "$ONNX_DIR/"
echo "DONE: ONNX export complete"
