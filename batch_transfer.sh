#!/bin/bash
# Batch transfer all ONNX files from wsl2-local to orin-nx (direct via Tailscale)
# Run on wsl2-local

SRC_DIR="/home/harve/trt-work/onnx-layers"
DST_HOST="100.82.225.102"
DST_DIR="/tmp"

# Transfer all layer files
for f in "$SRC_DIR"/layer_*_fp16.onnx; do
    name=$(basename "$f")
    dst="$DST_DIR/$name"

    if timeout 30 ssh -o ConnectTimeout=10 harvest@"$DST_HOST" "test -f $dst" 2>/dev/null; then
        echo "SKIP $name (already exists)"
        continue
    fi

    echo "TRANSFER $name -> orin-nx"
    scp -o ConnectTimeout=10 "$f" "harvest@$DST_HOST:$dst"
    if [ $? -eq 0 ]; then
        echo "  OK"
    else
        echo "  FAILED"
    fi
done

# Transfer pre/post components
for name in t_embedder_fp16.onnx prompt_preprocessor_fp16.onnx latent_preprocessor_fp16.onnx final_projection_fp16.onnx; do
    dst="$DST_DIR/$name"
    if timeout 30 ssh -o ConnectTimeout=10 harvest@"$DST_HOST" "test -f $dst" 2>/dev/null; then
        echo "SKIP $name (already exists)"
        continue
    fi
    echo "TRANSFER $name -> orin-nx"
    scp -o ConnectTimeout=10 "$SRC_DIR/$name" "harvest@$DST_HOST:$dst"
done

echo "DONE"
