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

This repo does not ship model weights, ONNX files, or TensorRT engines. You need
to download the upstream model and build engines for your target Jetson first.

For a host that already has the model and engines in the expected locations:

```bash
# 384x384 text-to-image, default 4 steps
RESOLUTION=384 scripts/run/run_3drope_basic_refiner.sh

# 512x512 text-to-image, default 4 steps
RESOLUTION=512 scripts/run/run_3drope_basic_refiner.sh
```

Output defaults:

```text
384: /home/harvest/z-image-output/output_384.png
512: /home/harvest/z-image-output/output_3drope.png
```

Override host paths if your layout differs:

```bash
MODEL_ROOT_HOST=/path/to/models \
ENGINE_DIR_384_HOST=/path/to/trt-engines-384-bf16 \
ENGINE_DIR_512_HOST=/path/to/trt-engines-bf16 \
OUTPUT_DIR_HOST=/path/to/output \
RESOLUTION=384 \
scripts/run/run_3drope_basic_refiner.sh
```

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

The high-level flow is:

```text
1. Download Tongyi-MAI/Z-Image-Turbo on an export machine
2. Export ONNX with scripts/export/export_all_layers_fp16.py
3. Export refiners with scripts/export/export_refiners.py
4. Build TensorRT engines on Jetson with trtexec --bf16
5. Run scripts/run/run_3drope_basic_refiner.sh
```

Example 384 export on a CUDA workstation:

```bash
RESOLUTION=384 \
OUTPUT_DIR=/home/user/trt-work/onnx-384 \
python3 scripts/export/export_all_layers_fp16.py

RESOLUTION=384 \
OUTPUT_DIR=/home/user/trt-work/onnx-384 \
FORCE_EXPORT=1 \
python3 scripts/export/export_refiners.py
```

Build each ONNX file on Jetson:

```bash
mkdir -p /home/user/models/axera-onnx/trt-engines-384-bf16
cd /home/user/trt-work/onnx-384

for f in *.onnx; do
  name="$(basename "$f" .onnx)"
  /usr/src/tensorrt/bin/trtexec \
    --onnx="$f" \
    --saveEngine="/home/user/models/axera-onnx/trt-engines-384-bf16/${name}.engine" \
    --bf16 \
    --timingCacheFile=/tmp/trt_cache.txt
done
```

See [docs/TRT_STATUS.md](docs/TRT_STATUS.md) for the full validated build and
performance notes.

## Architecture

```text
Text encoder (PyTorch Qwen3) -> TRT prompt preprocessor -> context refiners
Random or img2img latent -> TRT latent preprocessor -> noise refiners
Concatenate image/text tokens
30 x split TensorRT transformer layer engines
TRT final projection -> FlowMatch scheduler step
VAE decode (PyTorch)
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
- Docker image with PyTorch, diffusers, transformers, Pillow, and TensorRT Python bindings

Export machine:

- CUDA GPU with enough memory for ONNX export
- PyTorch 2.5+
- diffusers with Z-Image pipeline support
- ONNX

## Known Limits

- Orin Nano / 8GB is not validated.
- Engines are static-shape and resolution-specific.
- Model weights, ONNX files, and TensorRT engines are not included.
- Runtime still uses PyTorch for the text encoder and VAE.
- Docker launcher currently assumes Jetson-style host CUDA/TensorRT library mounts.

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
