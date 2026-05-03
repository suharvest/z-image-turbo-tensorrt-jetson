#!/bin/bash
set -euo pipefail

ONNX_DIR="/home/harvest/models/axera-onnx/transformer_onnx"
ENGINE_DIR="/home/harvest/models/axera-onnx/trt-engines"
LOG_DIR="/home/harvest/models/axera-onnx/logs"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
DOWNLOADER="/home/harvest/models/axera-onnx/onnx_download.py"

mkdir -p "$ONNX_DIR" "$ENGINE_DIR" "$LOG_DIR"

LOG="$LOG_DIR/batch_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date +%H:%M:%S)] $*"; }

download() {
    local fname="$1"
    local opath="$ONNX_DIR/$fname"
    if [ -f "$opath" ]; then
        local sz=$(stat -c%s "$opath" 2>/dev/null || echo 0)
        log "SKIP download: $fname (exists, $(numfmt --to=iec $sz))"
        return 0
    fi
    log "DOWNLOAD START: $fname"
    python3 "$DOWNLOADER" "$fname" && log "DOWNLOAD OK: $fname" || {
        log "DOWNLOAD FAILED: $fname"
        return 1
    }
    local sz=$(stat -c%s "$opath" 2>/dev/null || echo 0)
    log "DOWNLOAD SIZE: $fname = $(numfmt --to=iec $sz)"
}

build_engine() {
    local fname="$1"
    local opath="$ONNX_DIR/$fname"
    local epath="$ENGINE_DIR/${fname}.engine"
    local blog="$LOG_DIR/${fname}.build.log"

    if [ -f "$epath" ]; then
        local sz=$(stat -c%s "$epath" 2>/dev/null || echo 0)
        log "SKIP build: ${fname}.engine (exists, $(numfmt --to=iec $sz))"
        return 0
    fi

    if [ ! -f "$opath" ]; then
        log "BUILD SKIP: $fname (ONNX file not found)"
        return 1
    fi

    # Wait for any other trtexec to finish
    while pgrep -f trtexec > /dev/null 2>&1; do
        log "WAIT: another trtexec running, sleeping 10s..."
        sleep 10
    done

    log "BUILD START: $fname -> ${fname}.engine"
    local start_ts=$(date +%s)
    if "$TRTEXEC" --onnx="$opath" --saveEngine="$epath" --fp16 > "$blog" 2>&1; then
        local end_ts=$(date +%s)
        local sz=$(stat -c%s "$epath" 2>/dev/null || echo 0)
        local elapsed=$((end_ts - start_ts))
        log "BUILD OK: ${fname}.engine ($(numfmt --to=iec $sz), ${elapsed}s)"
        # Delete ONNX to save space
        rm -f "$opath"
        log "DELETED ONNX: $fname"
        return 0
    else
        local end_ts=$(date +%s)
        local elapsed=$((end_ts - start_ts))
        log "BUILD FAILED: $fname (rc=$?, ${elapsed}s, see $blog)"
        tail -20 "$blog"
        return 1
    fi
}

# Process: download + build + delete
process() {
    local fname="$1"
    TOTAL=$((TOTAL + 1))
    log "--- [$SUCCESS/$TOTAL] Processing: $fname ---"
    if download "$fname" && build_engine "$fname"; then
        SUCCESS=$((SUCCESS + 1))
    else
        FAILED=$((FAILED + 1))
        log "WARNING: failed on $fname, continuing..."
    fi
    log "Progress: $SUCCESS/$TOTAL done, $FAILED failed"
    log "Disk: $(df -h / | tail -1 | awk '{print $4}') free"
}

# ============================
# MAIN
# ============================

TOTAL=0
SUCCESS=0
FAILED=0

log "=========================================="
log "BATCH PROCESS START"
log "Disk before: $(df -h / | tail -1 | awk '{print $4}') free"
log "=========================================="

# Phase 1: auto_00 (5MB)
log "=== Phase 1: auto_00 (5MB) ==="
process "auto_00_model_layers_29_Add_4_output_0_to_sample_auto.onnx"

# Phase 2: cfg_04 through cfg_32 (29 files, 724MB each)
log "=== Phase 2: cfg_04..cfg_32 (29 files, ~724MB each) ==="
for nn in $(seq -w 4 32); do
    from=$((10#$nn - 4))
    to=$((10#$nn - 3))
    fname="cfg_${nn}_model_layers_${from}_Add_4_output_0_to_model_layers_${to}_Add_4_output_0_config.onnx"
    process "$fname"
done

# Phase 3: cfg_03 and cfg_00 (already downloaded, just build + delete)
log "=== Phase 3: Build already-downloaded cfg_03, cfg_00 ==="
process "cfg_03_model_Slice_1_output_0_to_model_layers_0_Add_4_output_0_config.onnx"
process "cfg_00_timestep_to_model_t_embedder_mlp_mlp_2_Gemm_output_0_config.onnx"

# Phase 4: cfg_01 (1.46GB)
log "=== Phase 4: cfg_01 (1.46GB) ==="
process "cfg_01_prompt_embeds_to_model_Slice_1_output_0_config.onnx"

# Phase 5: cfg_02 (1.46GB)
log "=== Phase 5: cfg_02 (1.46GB) ==="
process "cfg_02_latent_model_input_to_model_Slice_output_0_config.onnx"

# Cleanup old /tmp engine
if [ -f /tmp/cfg_03.engine ]; then
    rm -f /tmp/cfg_03.engine
    log "CLEANUP: removed /tmp/cfg_03.engine"
fi

log "=========================================="
log "BATCH PROCESS COMPLETE"
log "Total: $TOTAL, Success: $SUCCESS, Failed: $FAILED"
log "Disk after: $(df -h / | tail -1 | awk '{print $4}') free"
log "=========================================="
log "Engine files:"
ls -lh "$ENGINE_DIR/"
log "=========================================="
