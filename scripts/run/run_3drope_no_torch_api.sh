#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PATH="${API_PATH:-$SCRIPT_DIR/api_server_no_torch.py}"
PIPELINE_PATH="${PIPELINE_PATH:-$SCRIPT_DIR/pipeline_trt_no_torch.py}"
CODE_MOUNTS=()
if [ -f "$API_PATH" ]; then
  CODE_MOUNTS+=("-v" "$API_PATH:/workspace/api_server_no_torch.py:ro")
elif [ -n "${API_PATH:-}" ] && [ "$API_PATH" != "$SCRIPT_DIR/api_server_no_torch.py" ]; then
  echo "API path not found: $API_PATH" >&2
  exit 1
fi
if [ -f "$PIPELINE_PATH" ]; then
  CODE_MOUNTS+=("-v" "$PIPELINE_PATH:/workspace/pipeline_trt_no_torch.py:ro")
elif [ -n "${PIPELINE_PATH:-}" ] && [ "$PIPELINE_PATH" != "$SCRIPT_DIR/pipeline_trt_no_torch.py" ]; then
  echo "Pipeline path not found: $PIPELINE_PATH" >&2
  exit 1
fi

MODEL_ROOT_HOST="${MODEL_ROOT_HOST:-$HOME/models}"
ENGINE_DIR_512_HOST="${ENGINE_DIR_512_HOST:-$HOME/models/axera-onnx/trt-engines-bf16}"
ENGINE_DIR_384_HOST="${ENGINE_DIR_384_HOST:-$HOME/models/axera-onnx/trt-engines-384-bf16}"
TEXT_ENCODER_ENGINE_DIR_HOST="${TEXT_ENCODER_ENGINE_DIR_HOST:-$HOME/models/axera-onnx/trt-text-encoder-split-g4}"
OUTPUT_DIR_HOST="${OUTPUT_DIR_HOST:-$HOME/z-image-output}"
UPLOAD_DIR_HOST="${UPLOAD_DIR_HOST:-$HOME/z-image-input/api-uploads}"
DOCKER_IMAGE="${DOCKER_IMAGE:-z-image-jetson-no-torch:latest}"
MODEL_DIR="${MODEL_DIR:-/models/z-image-turbo-fp8-diffusers}"
CUDA_HOST="${CUDA_HOST:-/usr/local/cuda-12.6}"
TRT_PY_HOST="${TRT_PY_HOST:-/usr/lib/python3.10/dist-packages/tensorrt}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"

RESOLUTION="${RESOLUTION:-384}"
if [ "$RESOLUTION" = "384" ]; then
  ENGINE_DIR="/engines-384-bf16"
  ENGINE_DIR_HOST="$ENGINE_DIR_384_HOST"
elif [ "$RESOLUTION" = "512" ]; then
  ENGINE_DIR="/engines-bf16"
  ENGINE_DIR_HOST="$ENGINE_DIR_512_HOST"
else
  echo "Unsupported RESOLUTION=$RESOLUTION. Use 384 or 512." >&2
  exit 1
fi

for required_path in "$MODEL_ROOT_HOST" "$ENGINE_DIR_HOST" "$TEXT_ENCODER_ENGINE_DIR_HOST" "$CUDA_HOST" "$TRT_PY_HOST"; do
  if [ ! -e "$required_path" ]; then
    echo "Required path not found: $required_path" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR_HOST" "$UPLOAD_DIR_HOST"

docker run --rm --privileged --network=host \
  -v /usr/lib/aarch64-linux-gnu:/host-libs:ro \
  -v /etc/alternatives:/etc/alternatives:ro \
  -v "$CUDA_HOST:/usr/local/cuda:ro" \
  -v "$TRT_PY_HOST:/usr/local/trt-py:ro" \
  -v "$MODEL_ROOT_HOST:/models:ro" \
  -v "$ENGINE_DIR_HOST:$ENGINE_DIR:ro" \
  -v "$TEXT_ENCODER_ENGINE_DIR_HOST:/text-encoder-engines:ro" \
  -v "$OUTPUT_DIR_HOST:/output" \
  -v "$UPLOAD_DIR_HOST:/uploads" \
  "${CODE_MOUNTS[@]}" \
  -e RESOLUTION="$RESOLUTION" \
  -e MODEL_DIR="$MODEL_DIR" \
  -e ENGINE_DIR="$ENGINE_DIR" \
  -e TEXT_ENCODER_ENGINE_DIR="/text-encoder-engines" \
  -e VAE_ENGINE_DIR="$ENGINE_DIR" \
  -e OUTPUT_DIR="/output" \
  -e UPLOAD_DIR="/uploads" \
  -e MAX_CACHED_LAYERS="${MAX_CACHED_LAYERS:-}" \
  -e PYTHONPATH=/workspace:/usr/local/trt-py \
  -e LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:/usr/local/cuda/nvvm/lib64:/host-libs:/host-libs/tegra:/host-libs/openblas-pthread" \
  "$DOCKER_IMAGE" uvicorn api_server_no_torch:app --host "$API_HOST" --port "$API_PORT" --workers 1
