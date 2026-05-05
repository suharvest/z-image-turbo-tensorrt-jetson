#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_PATH="${PIPELINE_PATH:-$SCRIPT_DIR/pipeline_trt_no_torch.py}"
PIPELINE_MOUNT=()
if [ -f "$PIPELINE_PATH" ]; then
  PIPELINE_MOUNT=("-v" "$PIPELINE_PATH:/workspace/pipeline_trt_no_torch.py:ro")
elif [ -n "${PIPELINE_PATH:-}" ] && [ "$PIPELINE_PATH" != "$SCRIPT_DIR/pipeline_trt_no_torch.py" ]; then
  echo "Pipeline path not found: $PIPELINE_PATH" >&2
  exit 1
fi

MODEL_ROOT_HOST="${MODEL_ROOT_HOST:-$HOME/models}"
ENGINE_DIR_512_HOST="${ENGINE_DIR_512_HOST:-$HOME/models/axera-onnx/trt-engines-bf16}"
ENGINE_DIR_384_HOST="${ENGINE_DIR_384_HOST:-$HOME/models/axera-onnx/trt-engines-384-bf16}"
TEXT_ENCODER_ENGINE_DIR_HOST="${TEXT_ENCODER_ENGINE_DIR_HOST:-$HOME/models/axera-onnx/trt-text-encoder-split-g4}"
OUTPUT_DIR_HOST="${OUTPUT_DIR_HOST:-$HOME/z-image-output}"
DOCKER_IMAGE="${DOCKER_IMAGE:-z-image-jetson-no-torch:latest}"
MODEL_DIR="${MODEL_DIR:-/models/z-image-turbo-fp8-diffusers}"
CUDA_HOST="${CUDA_HOST:-/usr/local/cuda-12.6}"
TRT_PY_HOST="${TRT_PY_HOST:-/usr/lib/python3.10/dist-packages/tensorrt}"

RESOLUTION="${RESOLUTION:-384}"
if [ "$RESOLUTION" = "384" ]; then
  ENGINE_DIR="/engines-384-bf16"
  ENGINE_DIR_HOST="$ENGINE_DIR_384_HOST"
  OUTPUT_PATH="${OUTPUT_PATH:-/output/output_384_no_torch.png}"
elif [ "$RESOLUTION" = "512" ]; then
  ENGINE_DIR="/engines-bf16"
  ENGINE_DIR_HOST="$ENGINE_DIR_512_HOST"
  OUTPUT_PATH="${OUTPUT_PATH:-/output/output_512_no_torch.png}"
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

mkdir -p "$OUTPUT_DIR_HOST"

EXTRA_MOUNTS=()
INIT_LATENT_PATH="${INIT_LATENT_PATH:-}"
if [ -n "${INPUT_IMAGE_PATH:-}" ]; then
  if [ ! -f "$INPUT_IMAGE_PATH" ]; then
    echo "Input image not found: $INPUT_IMAGE_PATH" >&2
    exit 1
  fi
  EXTRA_MOUNTS+=("-v" "$INPUT_IMAGE_PATH:/input/init_image:ro")
  INIT_IMAGE="/input/init_image"
  if [ "${IMG2IMG_TWO_STAGE:-0}" = "1" ]; then
    INIT_LATENT_PATH="${INIT_LATENT_PATH:-/output/init_latent_no_torch.npz}"
  fi
fi

run_container() {
  local encode_only="$1"
  docker run --rm --privileged --network=host \
    -v /usr/lib/aarch64-linux-gnu:/host-libs:ro \
    -v /etc/alternatives:/etc/alternatives:ro \
    -v "$CUDA_HOST:/usr/local/cuda:ro" \
    -v "$TRT_PY_HOST:/usr/local/trt-py:ro" \
    -v "$MODEL_ROOT_HOST:/models:ro" \
    -v "$ENGINE_DIR_HOST:$ENGINE_DIR:ro" \
    -v "$TEXT_ENCODER_ENGINE_DIR_HOST:/text-encoder-engines:ro" \
    -v "$OUTPUT_DIR_HOST:/output" \
    "${PIPELINE_MOUNT[@]}" \
    "${EXTRA_MOUNTS[@]}" \
    -e RESOLUTION="$RESOLUTION" \
    -e MODEL_DIR="$MODEL_DIR" \
    -e ENGINE_DIR="$ENGINE_DIR" \
    -e TEXT_ENCODER_ENGINE_DIR="/text-encoder-engines" \
    -e VAE_ENGINE_DIR="$ENGINE_DIR" \
    -e OUTPUT_PATH="$OUTPUT_PATH" \
    -e NUM_STEPS="${NUM_STEPS:-4}" \
    -e MAX_CACHED_LAYERS="${MAX_CACHED_LAYERS:-}" \
    -e INIT_IMAGE="${INIT_IMAGE:-}" \
    -e INIT_LATENT_PATH="$INIT_LATENT_PATH" \
    -e IMG2IMG_ENCODE_ONLY="$encode_only" \
    -e STRENGTH="${STRENGTH:-0.6}" \
    -e PROMPT="${PROMPT:-A cute orange tabby cat sitting on a sunny windowsill, soft natural lighting, photorealistic, high detail}" \
    -e PYTHONPATH=/usr/local/trt-py \
    -e LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:/usr/local/cuda/nvvm/lib64:/host-libs:/host-libs/tegra:/host-libs/openblas-pthread" \
    "$DOCKER_IMAGE" python3 -u /workspace/pipeline_trt_no_torch.py
}

if [ -n "${INPUT_IMAGE_PATH:-}" ] && [ "${IMG2IMG_TWO_STAGE:-0}" = "1" ]; then
  echo "Preparing no-torch img2img latent..."
  run_container 1
  INIT_IMAGE=""
fi

run_container 0
