#!/usr/bin/env python3
"""Download ONNX file from HF mirror."""
import os
import sys

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import hf_hub_download

filename = sys.argv[1]
outdir = sys.argv[2] if len(sys.argv) > 2 else '/home/harvest/models/axera-onnx'

path = hf_hub_download(
    'AXERA-TECH/Z-Image-Turbo',
    f'transformer_onnx/{filename}',
    local_dir=outdir,
    local_dir_use_symlinks=False
)
print(f'OK: {path}')
