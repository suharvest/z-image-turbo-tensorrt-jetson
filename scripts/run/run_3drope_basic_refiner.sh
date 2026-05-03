#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_PATH="${PIPELINE_PATH:-$SCRIPT_DIR/pipeline_trt_v2.py}"

MODEL_ROOT_HOST="${MODEL_ROOT_HOST:-$HOME/models}"
ENGINE_DIR_512_HOST="${ENGINE_DIR_512_HOST:-$HOME/models/axera-onnx/trt-engines-bf16}"
ENGINE_DIR_384_HOST="${ENGINE_DIR_384_HOST:-$HOME/models/axera-onnx/trt-engines-384-bf16}"
OUTPUT_DIR_HOST="${OUTPUT_DIR_HOST:-$HOME/z-image-output}"
DOCKER_IMAGE="${DOCKER_IMAGE:-z-image-jetson:latest}"
MODEL_DIR="${MODEL_DIR:-/models/z-image-turbo-fp8-diffusers}"
CUDA_HOST="${CUDA_HOST:-/usr/local/cuda-12.6}"
TRT_PY_HOST="${TRT_PY_HOST:-/usr/lib/python3.10/dist-packages/tensorrt}"
NVIDIA_PIP_HOST="${NVIDIA_PIP_HOST:-$HOME/.local/lib/python3.10/site-packages/nvidia}"

RESOLUTION="${RESOLUTION:-512}"
if [ -z "${ENGINE_DIR:-}" ]; then
  if [ "$RESOLUTION" = "384" ]; then
    ENGINE_DIR="/engines-384-bf16"
  else
    ENGINE_DIR="/engines-bf16"
  fi
fi
if [ -z "${OUTPUT_PATH:-}" ]; then
  if [ "$RESOLUTION" = "384" ]; then
    OUTPUT_PATH="/output/output_384.png"
  else
    OUTPUT_PATH="/output/output_3drope.png"
  fi
fi
if [ -z "${MAX_CACHED_LAYERS:-}" ]; then
  if [ "$RESOLUTION" = "384" ]; then
    MAX_CACHED_LAYERS="23"
  else
    MAX_CACHED_LAYERS="18"
  fi
fi
if [ -z "${NUM_STEPS:-}" ]; then
  if [ "$RESOLUTION" = "384" ]; then
    NUM_STEPS="4"
  else
    NUM_STEPS="4"
  fi
fi

if [ "$RESOLUTION" != "384" ] && [ "$RESOLUTION" != "512" ]; then
  echo "Unsupported RESOLUTION=$RESOLUTION. Use 384 or 512." >&2
  exit 1
fi

if [ ! -f "$PIPELINE_PATH" ]; then
  echo "Pipeline script not found: $PIPELINE_PATH" >&2
  exit 1
fi

if [ ! -d "$MODEL_ROOT_HOST" ]; then
  echo "Model root not found: $MODEL_ROOT_HOST" >&2
  echo "Set MODEL_ROOT_HOST to the host directory that contains the Z-Image model." >&2
  exit 1
fi

if [ "$RESOLUTION" = "384" ]; then
  ENGINE_DIR_HOST="$ENGINE_DIR_384_HOST"
else
  ENGINE_DIR_HOST="$ENGINE_DIR_512_HOST"
fi
if [ ! -d "$ENGINE_DIR_HOST" ]; then
  echo "TensorRT engine directory not found for ${RESOLUTION}p mode: $ENGINE_DIR_HOST" >&2
  echo "Build engines first or set ENGINE_DIR_${RESOLUTION}_HOST." >&2
  exit 1
fi

for required_mount in "$CUDA_HOST" "$TRT_PY_HOST" "$NVIDIA_PIP_HOST"; do
  if [ ! -d "$required_mount" ]; then
    echo "Required host mount not found: $required_mount" >&2
    echo "Set CUDA_HOST, TRT_PY_HOST, or NVIDIA_PIP_HOST for this Jetson environment." >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR_HOST"

EXTRA_MOUNTS=()
if [ -n "${INPUT_IMAGE_PATH:-}" ]; then
  if [ ! -f "$INPUT_IMAGE_PATH" ]; then
    echo "Input image not found: $INPUT_IMAGE_PATH" >&2
    exit 1
  fi
  EXTRA_MOUNTS+=("-v" "$INPUT_IMAGE_PATH:/input/init_image:ro")
  INIT_IMAGE="/input/init_image"
fi

docker run --rm --privileged --network=host \
  -v /usr/lib/aarch64-linux-gnu:/host-libs:ro \
  -v /etc/alternatives:/etc/alternatives:ro \
  -v "$CUDA_HOST:/usr/local/cuda:ro" \
  -v "$NVIDIA_PIP_HOST:/usr/local/nvidia-pip:ro" \
  -v "$TRT_PY_HOST:/usr/local/trt-py:ro" \
  -v "$MODEL_ROOT_HOST:/models:ro" \
  -v "$ENGINE_DIR_512_HOST:/engines-bf16:ro" \
  -v "$ENGINE_DIR_384_HOST:/engines-384-bf16:ro" \
  -v "$OUTPUT_DIR_HOST:/output" \
  -v "$PIPELINE_PATH:/workspace/test.py:ro" \
  "${EXTRA_MOUNTS[@]}" \
  -e RESOLUTION="$RESOLUTION" \
  -e ENGINE_DIR="$ENGINE_DIR" \
  -e MODEL_DIR="$MODEL_DIR" \
  -e OUTPUT_PATH="$OUTPUT_PATH" \
  -e PROMPT="${PROMPT:-A cute orange tabby cat sitting on a sunny windowsill, soft natural lighting, photorealistic, high detail}" \
  -e INIT_IMAGE="${INIT_IMAGE:-}" \
  -e STRENGTH="${STRENGTH:-0.6}" \
  -e USE_TRT_VAE="${USE_TRT_VAE:-0}" \
  -e VAE_ENGINE_DIR="${VAE_ENGINE_DIR:-$ENGINE_DIR}" \
  -e DEBUG_TENSOR_STATS="${DEBUG_TENSOR_STATS:-0}" \
  -e MINIMAL_PYTORCH_LOAD="${MINIMAL_PYTORCH_LOAD:-1}" \
  -e DELAY_LOAD_VAE="${DELAY_LOAD_VAE:-1}" \
  -e FREE_TRT_BEFORE_VAE="${FREE_TRT_BEFORE_VAE:-1}" \
  -e REUSE_LAYER_OUTPUT_BUFFERS="${REUSE_LAYER_OUTPUT_BUFFERS:-1}" \
  -e USE_GROUP_LAYERS="${USE_GROUP_LAYERS:-0}" \
  -e GROUP_00_04_ENGINE="${GROUP_00_04_ENGINE:-/models/axera-onnx/group-layers-00-04/layers_00_04_fp16.engine}" \
  -e MAX_CACHED_LAYERS="$MAX_CACHED_LAYERS" \
  -e NUM_STEPS="$NUM_STEPS" \
  -e PYTHONPATH=/usr/local/trt-py \
  -e LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:/usr/local/cuda/nvvm/lib64:/host-libs:/host-libs/tegra:/host-libs/openblas-pthread:/usr/local/nvidia-pip/cusparselt/lib" \
  "$DOCKER_IMAGE" python3 -u /workspace/test.py
