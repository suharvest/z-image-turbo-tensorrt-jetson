#!/bin/bash
# Build TRT engines on orin-nx for all Z-Image-Turbo ONNX layers
# Usage: run this on orin-nx

ONNX_DIR="/tmp"
ENGINE_DIR="/tmp/trt-engines"
LOG_FILE="/tmp/trt_build.log"
CACHE="/tmp/trt_cache.txt"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"

mkdir -p "$ENGINE_DIR"

echo "=== TRT Engine Build Started $(date) ===" | tee -a "$LOG_FILE"

build_one() {
    local onnx="$1"
    local name=$(basename "$onnx" .onnx)
    local engine="$ENGINE_DIR/${name}.engine"

    echo "[$(date +%H:%M:%S)] Building $name ..." | tee -a "$LOG_FILE"

    if $TRTEXEC \
        --onnx="$onnx" \
        --saveEngine="$engine" \
        --fp16 \
        --timingCacheFile="$CACHE" \
        >> "$LOG_FILE" 2>&1; then

        local size=$(du -h "$engine" | cut -f1)
        echo "[$(date +%H:%M:%S)]   OK: $engine ($size)" | tee -a "$LOG_FILE"
        return 0
    else
        echo "[$(date +%H:%M:%S)]   FAILED: $name" | tee -a "$LOG_FILE"
        return 1
    fi
}

# Build all layer ONNX files
count=0
failed=0
for onnx in "$ONNX_DIR"/layer_*_fp16.onnx; do
    if build_one "$onnx"; then
        count=$((count + 1))
    else
        failed=$((failed + 1))
    fi
done

# Build pre/post components
for onnx in "$ONNX_DIR"/t_embedder_fp16.onnx "$ONNX_DIR"/prompt_preprocessor_fp16.onnx "$ONNX_DIR"/latent_preprocessor_fp16.onnx "$ONNX_DIR"/final_projection_fp16.onnx; do
    if [ -f "$onnx" ]; then
        if build_one "$onnx"; then
            count=$((count + 1))
        else
            failed=$((failed + 1))
        fi
    fi
done

echo "=== Build Complete $(date) ===" | tee -a "$LOG_FILE"
echo "Success: $count, Failed: $failed" | tee -a "$LOG_FILE"
ls -lh "$ENGINE_DIR"/ | tee -a "$LOG_FILE"
