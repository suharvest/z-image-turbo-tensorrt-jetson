#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_PATH="${PIPELINE_PATH:-$SCRIPT_DIR/pipeline_trt_v2.py}"
if [ ! -f "$PIPELINE_PATH" ] && [ -f /tmp/pipe_3drope.py ]; then
  PIPELINE_PATH="/tmp/pipe_3drope.py"
fi

MODEL_ROOT_HOST="${MODEL_ROOT_HOST:-/home/harvest/models}"
ENGINE_DIR_512_HOST="${ENGINE_DIR_512_HOST:-/home/harvest/models/axera-onnx/trt-engines-bf16}"
ENGINE_DIR_384_HOST="${ENGINE_DIR_384_HOST:-/home/harvest/models/axera-onnx/trt-engines-384-bf16}"
OUTPUT_DIR_HOST="${OUTPUT_DIR_HOST:-/home/harvest/z-image-output}"

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
EXTRA_MOUNTS=()
if [ -n "${INPUT_IMAGE_PATH:-}" ]; then
  EXTRA_MOUNTS+=("-v" "$INPUT_IMAGE_PATH:/input/init_image:ro")
  INIT_IMAGE="/input/init_image"
fi

docker run --rm --privileged --network=host \
  -v /usr/lib/aarch64-linux-gnu:/host-libs:ro \
  -v /etc/alternatives:/etc/alternatives:ro \
  -v /usr/local/cuda-12.6:/usr/local/cuda:ro \
  -v /home/harvest/.local/lib/python3.10/site-packages/nvidia:/usr/local/nvidia-pip:ro \
  -v /usr/lib/python3.10/dist-packages/tensorrt:/usr/local/trt-py:ro \
  -v "$MODEL_ROOT_HOST:/models:ro" \
  -v "$ENGINE_DIR_512_HOST:/engines-bf16:ro" \
  -v "$ENGINE_DIR_384_HOST:/engines-384-bf16:ro" \
  -v "$OUTPUT_DIR_HOST:/output" \
  -v "$PIPELINE_PATH:/workspace/test.py:ro" \
  "${EXTRA_MOUNTS[@]}" \
  -e RESOLUTION="$RESOLUTION" \
  -e ENGINE_DIR="$ENGINE_DIR" \
  -e OUTPUT_PATH="$OUTPUT_PATH" \
  -e PROMPT="${PROMPT:-A cute orange tabby cat sitting on a sunny windowsill, soft natural lighting, photorealistic, high detail}" \
  -e INIT_IMAGE="${INIT_IMAGE:-}" \
  -e STRENGTH="${STRENGTH:-0.6}" \
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
  z-image-jetson:latest python3 -u /workspace/test.py
