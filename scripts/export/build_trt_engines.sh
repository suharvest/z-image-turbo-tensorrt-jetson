#!/usr/bin/env bash
set -euo pipefail

ONNX_DIR="${ONNX_DIR:-/path/to/onnx-384}"
ENGINE_DIR="${ENGINE_DIR:-/path/to/trt-engines-384-bf16}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
TIMING_CACHE="${TIMING_CACHE:-/tmp/trt_cache.txt}"
PRECISION="${PRECISION:-bf16}"
LOG_FILE="${LOG_FILE:-$ENGINE_DIR/build.log}"

if [ ! -d "$ONNX_DIR" ]; then
  echo "ONNX_DIR does not exist: $ONNX_DIR" >&2
  exit 1
fi
if [ ! -x "$TRTEXEC" ]; then
  echo "trtexec not found or not executable: $TRTEXEC" >&2
  exit 1
fi
if [ "$PRECISION" != "bf16" ]; then
  echo "Only PRECISION=bf16 is validated for Z-Image-Turbo on Jetson." >&2
  exit 1
fi

mkdir -p "$ENGINE_DIR"
echo "=== TensorRT BF16 build started $(date) ===" | tee "$LOG_FILE"
echo "ONNX_DIR=$ONNX_DIR" | tee -a "$LOG_FILE"
echo "ENGINE_DIR=$ENGINE_DIR" | tee -a "$LOG_FILE"

count=0
failed=0
for onnx in "$ONNX_DIR"/*.onnx; do
  [ -e "$onnx" ] || {
    echo "No .onnx files found in $ONNX_DIR" >&2
    exit 1
  }
  name="$(basename "$onnx" .onnx)"
  engine="$ENGINE_DIR/${name}.engine"
  echo "[$(date +%H:%M:%S)] Building $name" | tee -a "$LOG_FILE"
  if "$TRTEXEC" \
      --onnx="$onnx" \
      --saveEngine="$engine" \
      --bf16 \
      --timingCacheFile="$TIMING_CACHE" \
      >> "$LOG_FILE" 2>&1; then
    du -h "$engine" | tee -a "$LOG_FILE"
    count=$((count + 1))
  else
    echo "FAILED: $name" | tee -a "$LOG_FILE"
    failed=$((failed + 1))
  fi
done

echo "=== TensorRT BF16 build finished $(date) ===" | tee -a "$LOG_FILE"
echo "Success: $count, failed: $failed" | tee -a "$LOG_FILE"
if [ "$failed" -ne 0 ]; then
  exit 1
fi
