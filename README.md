# Z-Image-Turbo TensorRT for Jetson

Z-Image-Turbo on Jetson Orin NX: **384px in ~73s**, **512px in ~100s** with TensorRT BF16.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Jetson Orin NX](https://img.shields.io/badge/Jetson-Orin%20NX-green)
![TensorRT](https://img.shields.io/badge/TensorRT-BF16-blue)
![Z--Image--Turbo](https://img.shields.io/badge/Z--Image--Turbo-6B-purple)

This repository contains a reproducible ONNX -> TensorRT pipeline for running
[Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
locally on NVIDIA Jetson Orin NX. It exports the Z-Image transformer into
static-shape BF16 TensorRT engines, runs text-to-image and img2img inference,
and documents the memory/performance tradeoffs needed to fit a 6B image model
on a 16GB edge device.

> Status: validated on **Jetson Orin NX 16GB**. Orin Nano / 8GB devices are not
> validated yet and should be treated as experimental.

## Demo

| Text-to-image, 384x384 | Img2img + prompt, 384x384 |
|---|---|
| ![384 text-to-image cat](media/text2img-384.png) | ![384 img2img red scarf cat](media/img2img-red-scarf-384.png) |

The img2img example uses the left image as a reference and the prompt
`wearing a small red scarf around its neck`.

## What Works

- Text-to-image inference at 384x384 and 512x512
- Z-Image img2img via VAE encode + FlowMatch noise scaling
- Static-shape TensorRT BF16 engines for 384 and 512 modes
- Layer-engine cache tuning for Jetson Orin NX 16GB
- Basic-mode `noise_refiner` and `context_refiner` parity with diffusers
- Export scripts for transformer layers, refiners, and pre/post processors

## Performance

Measured on Jetson Orin NX 16GB, JetPack 6, TensorRT 10.3, BF16 engines.

| Mode | Default steps | Total time | TRT denoise | Notes |
|---|---:|---:|---:|---|
| 384 text-to-image | 4 | 73.2s | 37.1s | Best speed/quality balance |
| 384 text-to-image | 8 | 107.8s | 71.7s | Highest validated 384 quality |
| 512 text-to-image | 4 | 100.2s | 63.1s | Best speed/quality balance |
| 512 text-to-image | 8 | 159.7s | 123.0s | Highest validated 512 run |
| 384 img2img | 8, strength 0.65 | 83.9s | 46.0s | 5 effective denoise steps |

Cache limits on Orin NX 16GB:

| Resolution | Default cache | OOM boundary |
|---|---:|---|
| 384 | 23 layers | 24 layers OOM during step 1 |
| 512 | 18 layers | 19 layers OOM during step 1 |

## Quickstart

This repo does not commit model weights, ONNX files, or TensorRT engines to
normal git because each resolution is roughly 12GB of generated artifacts. You
can reproduce them from the public upstream model, or publish/download them from
a separate Hugging Face/Git LFS artifact repo. See
[docs/ARTIFACTS.md](docs/ARTIFACTS.md).

For a host that already has the model and engines in the expected locations:

```bash
# 384x384 text-to-image, default 4 steps
RESOLUTION=384 scripts/run/run_3drope_basic_refiner.sh

# 512x512 text-to-image, default 4 steps
RESOLUTION=512 scripts/run/run_3drope_basic_refiner.sh
```

Output defaults:

```text
384: $HOME/z-image-output/output_384.png
512: $HOME/z-image-output/output_3drope.png
```

Override host paths if your layout differs:

```bash
DOCKER_IMAGE=z-image-jetson:latest \
MODEL_ROOT_HOST=/path/to/models \
MODEL_DIR=/models/z-image-turbo-fp8-diffusers \
ENGINE_DIR_384_HOST=/path/to/trt-engines-384-bf16 \
ENGINE_DIR_512_HOST=/path/to/trt-engines-bf16 \
OUTPUT_DIR_HOST=/path/to/output \
CUDA_HOST=/usr/local/cuda-12.6 \
TRT_PY_HOST=/usr/lib/python3.10/dist-packages/tensorrt \
NVIDIA_PIP_HOST=/path/to/python/site-packages/nvidia \
RESOLUTION=384 \
scripts/run/run_3drope_basic_refiner.sh
```

`MODEL_ROOT_HOST` is mounted into the container as `/models`. The default
`MODEL_DIR` expects the diffusers model at `/models/z-image-turbo-fp8-diffusers`.
Set `MODEL_DIR` if your model folder has a different name.

## Img2Img

```bash
RESOLUTION=384 \
NUM_STEPS=8 \
STRENGTH=0.65 \
INPUT_IMAGE_PATH=/path/to/reference.png \
OUTPUT_PATH=/output/output_img2img.png \
PROMPT="A cute orange tabby cat wearing a small red scarf, photorealistic" \
scripts/run/run_3drope_basic_refiner.sh
```

`STRENGTH` controls how much the reference image changes:

- `0.3-0.5`: preserve composition strongly, small edits
- `0.6-0.7`: good prompt/edit balance
- `0.8+`: larger semantic changes, more drift

## Repository Layout

```text
scripts/run/       Runtime pipeline and Docker launcher
scripts/export/    ONNX export and TensorRT build helpers
scripts/bench/     Python vs C++ TensorRT call-overhead probes
scripts/debug/     Layer comparison, RoPE, scale, and validation probes
scripts/legacy/    Earlier conversion and baseline scripts kept for reference
docs/              Detailed status, handoff notes, and optimization history
media/             README demo images
```

## Build Engines

The high-level flow is fully reproducible from the public upstream model:

```text
1. Download Tongyi-MAI/Z-Image-Turbo on an export machine
2. Export ONNX with scripts/export/export_all_layers_fp16.py
3. Export refiners with scripts/export/export_refiners.py
4. Build TensorRT engines on Jetson with trtexec --bf16
5. Run scripts/run/run_3drope_basic_refiner.sh
```

Example 384 export on a CUDA workstation:

```bash
MODEL_PATH=Tongyi-MAI/Z-Image-Turbo \
TRANSFORMER_SUBFOLDER=transformer \
RESOLUTION=384 \
OUTPUT_DIR=/home/user/trt-work/onnx-384 \
python3 scripts/export/export_all_layers_fp16.py

MODEL_PATH=Tongyi-MAI/Z-Image-Turbo \
TRANSFORMER_SUBFOLDER=transformer \
RESOLUTION=384 \
OUTPUT_DIR=/home/user/trt-work/onnx-384 \
FORCE_EXPORT=1 \
python3 scripts/export/export_refiners.py
```

Build each ONNX file on Jetson:

```bash
ONNX_DIR=/home/user/trt-work/onnx-384 \
ENGINE_DIR=/home/user/models/axera-onnx/trt-engines-384-bf16 \
scripts/export/build_trt_engines.sh
```

Optional VAE TensorRT export:

```bash
MODEL_PATH=Tongyi-MAI/Z-Image-Turbo \
RESOLUTION=384 \
OUTPUT_DIR=/home/user/trt-work/onnx-384 \
python3 scripts/export/export_vae.py

ONNX_DIR=/home/user/trt-work/onnx-384 \
ENGINE_DIR=/home/user/models/axera-onnx/trt-engines-384-bf16 \
scripts/export/build_trt_engines.sh
```

Keep the default BF16 TensorRT build for VAE engines. The VAE ONNX filenames
include `fp16` because the exported tensors are FP16, but the decoder engine
produced NaNs on Orin NX when built with `trtexec --fp16`.

Run with TRT VAE:

```bash
USE_TRT_VAE=1 RESOLUTION=384 scripts/run/run_3drope_basic_refiner.sh
```

Optional text encoder TensorRT export. The monolithic exporter is kept for
experimentation, but the single 36-layer Qwen3 text encoder ONNX OOMs the
TensorRT builder on Orin NX. Use the split exporter for Jetson:

```bash
MODEL_PATH=Tongyi-MAI/Z-Image-Turbo \
OUTPUT_DIR=/home/user/trt-work/onnx-text-encoder-split-g4 \
GROUP_SIZE=4 \
python3 scripts/export/export_text_encoder_split.py

for onnx in /home/user/trt-work/onnx-text-encoder-split-g4/*.onnx; do
  base="$(basename "$onnx" .onnx)"
  /usr/src/tensorrt/bin/trtexec \
    --onnx="$onnx" \
    --saveEngine="/home/user/models/axera-onnx/trt-text-encoder-split-g4/${base}.engine" \
    --bf16 \
    --builderOptimizationLevel=0 \
    --memPoolSize=workspace:1024 \
    --skipInference
done
```

Run with TRT text encoder:

```bash
USE_TRT_TEXT_ENCODER=1 \
TEXT_ENCODER_ENGINE_DIR=/models/axera-onnx/trt-text-encoder-split-g4 \
TEXT_ENCODER_GROUPS=0-3,4-7,8-11,12-15,16-19,20-23,24-27,28-31,32-35 \
RESOLUTION=384 \
scripts/run/run_3drope_basic_refiner.sh
```

See [docs/TRT_STATUS.md](docs/TRT_STATUS.md) for the full validated build and
performance notes, and [docs/ARTIFACTS.md](docs/ARTIFACTS.md) for suggested
ONNX/engine release layout.

For a newcomer-oriented checklist, see [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

## Architecture

```text
Text encoder (PyTorch fallback or TensorRT) -> TRT prompt preprocessor -> context refiners
Random or img2img latent -> TRT latent preprocessor -> noise refiners
Concatenate image/text tokens
30 x split TensorRT transformer layer engines
TRT final projection -> FlowMatch scheduler step
VAE decode (PyTorch fallback or TensorRT with USE_TRT_VAE=1)
```

Why BF16:

- FP16 attention overflow made realistic layers behave incorrectly.
- BF16 keeps the exponent range needed by the Z-Image transformer.
- AdaLN modulation is kept in FP32 during ONNX export for stability.

Why split engines:

- A 5-layer group engine was technically viable but consumed enough context
  memory to reduce the layer cache budget and ended up slower on Orin NX 16GB.
- Split engines plus layer caching were faster for the validated 384/512 modes.

## Requirements

Validated target:

- NVIDIA Jetson Orin NX 16GB
- JetPack 6 / Ubuntu 22.04
- TensorRT 10.3
- CUDA 12.6 host libraries
- Docker image with PyTorch, tokenizers, Pillow, and TensorRT Python bindings

The Docker image is only the runtime environment. It does not contain model
weights, ONNX files, TensorRT engines, or generated outputs. `docker/Dockerfile.jetson`
is provided as a starting template, but Jetson PyTorch wheels depend on your
JetPack/CUDA version. If you already have a working Jetson PyTorch image, set
`DOCKER_IMAGE` and use that instead. `diffusers` and `transformers` are only
needed for PyTorch fallback paths and export tools; the validated TRT text
encoder + TRT VAE runtime uses the lightweight tokenizer and built-in FlowMatch
scheduler.

Example image build flow:

```bash
docker build \
  --build-arg TORCH_WHEEL_URL=https://example.com/path/to/jetson-torch.whl \
  -f docker/Dockerfile.jetson \
  -t z-image-jetson:latest .
```

Export machine:

- CUDA GPU with enough memory for ONNX export
- PyTorch 2.5+
- diffusers with Z-Image pipeline support
- ONNX

## Known Limits

- Orin Nano / 8GB is not validated.
- Engines are static-shape and resolution-specific.
- Model weights, ONNX files, and TensorRT engines are not committed to normal git.
- Runtime still uses PyTorch tensors as CUDA buffers. The default runtime
  scheduler is a built-in FlowMatch Euler implementation; set
  `USE_DIFFUSERS_SCHEDULER=1` only if you need to compare against diffusers.
- With `USE_TRT_TEXT_ENCODER=1`, the default tokenizer path uses `tokenizers`
  directly. Set `USE_TRANSFORMERS_TOKENIZER=1` only if you need to compare
  against transformers.
- The VAE and text encoder can be moved to TensorRT with `USE_TRT_VAE=1` and
  `USE_TRT_TEXT_ENCODER=1`; on Jetson, use the split text encoder engines rather
  than the monolithic text encoder.
- Docker launcher currently assumes Jetson-style host CUDA/TensorRT library mounts.
- If `USE_TRT_VAE=1` produces a black image, rebuild `vae_decoder_fp16.onnx`
  with BF16 and rerun with `DEBUG_TENSOR_STATS=1` to check for NaNs.

## License

Repository code is released under the [MIT License](LICENSE).

The upstream `Tongyi-MAI/Z-Image-Turbo` model is licensed separately under
Apache-2.0. Model weights, derived ONNX exports, and derived TensorRT engines
are not redistributed in this repository; follow the upstream model license and
terms when downloading or distributing those artifacts.

## Acknowledgements

- [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
  for the base model.
- [Hugging Face diffusers](https://github.com/huggingface/diffusers) for the
  Z-Image pipeline implementation.
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) for engine build and
  runtime inference.
