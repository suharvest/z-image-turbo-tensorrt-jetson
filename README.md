# Z-Image-Turbo TensorRT for Jetson

Small image. Fast edge inference. Simple Jetson deployment for Z-Image-Turbo.

Run the 6B [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
image model locally on NVIDIA Jetson Orin NX with prebuilt TensorRT artifacts,
a **428MB** runtime image, and a ready-to-use HTTP API.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Jetson Orin NX](https://img.shields.io/badge/Jetson-Orin%20NX-green)
![TensorRT](https://img.shields.io/badge/TensorRT-BF16-blue)
![Z--Image--Turbo](https://img.shields.io/badge/Z--Image--Turbo-6B-purple)
![Runtime Image](https://img.shields.io/badge/runtime-428MB-brightgreen)

This project packages the hard parts of edge image generation:

- **Small runtime**: published no-PyTorch Docker image is about **428MB**.
- **High edge performance**: fastest validated path reaches **384px in ~73s** and **512px in ~100s** on Orin NX 16GB.
- **Simple deployment**: pull the runtime image, download HF artifacts, start the API.
- **Text-to-image and img2img**: generate from text or edit a reference image with a prompt.
- **Reproducible conversion path**: export ONNX and rebuild TensorRT engines when you need to.

> Status: validated on **Jetson Orin NX 16GB**, JetPack 6, TensorRT 10.3.
> Orin Nano / 8GB devices are not validated yet and should be treated as experimental.

## Why This Repo

Running a large image model on Jetson usually means a large Python/PyTorch
environment, heavy memory pressure, and a long model-conversion trail. This repo
focuses on the deployable result:

| Goal | Result |
|---|---|
| Small runtime image | `428MB` no-PyTorch Jetson image |
| High edge performance | fastest path: `384x384` in ~73s, `512x512` in ~100s |
| Simple deployment | Docker launcher + HTTP API |
| Reproducible artifacts | HF-hosted TensorRT engines and local rebuild scripts |

## Demo

| Text-to-image, 384x384 | Img2img + prompt, 384x384 |
|---|---|
| ![384 text-to-image cat](media/text2img-384.png) | ![384 img2img red scarf cat](media/img2img-red-scarf-384.png) |

The img2img example uses the left image as a reference and the prompt
`wearing a small red scarf around its neck`.

## What You Get

- Ready-to-run HTTP API for text-to-image and img2img.
- Published Jetson runtime image: `sensecraft-missionpack.seeed.cn/solution/z-image-jetson-no-torch:latest`.
- Published TensorRT engine artifacts on Hugging Face:
  `harvestsu/z-image-turbo-jetson-trt-artifacts`.
- Static-shape BF16 TensorRT engines for 384 and 512 modes.
- Layer-engine cache tuning for Jetson Orin NX 16GB.
- Export scripts for transformer layers, refiners, VAE, and text encoder.
- Engineering notes for failures, memory limits, and correctness checks.

## Performance

Measured on Jetson Orin NX 16GB, JetPack 6, TensorRT 10.3, BF16 engines.

There are two useful runtime profiles:

| Profile | Best for | Runtime image | 512 text-to-image |
|---|---|---:|---:|
| Fastest validated path | Benchmarking / maximum speed | Larger PyTorch-capable image | ~100.2s |
| Slim deployment path | Small image + HTTP API | ~428MB no-PyTorch image | ~117.4s |

| Mode | Default steps | Total time | TRT denoise | Notes |
|---|---:|---:|---:|---|
| 384 text-to-image | 4 | 73.2s | 37.1s | Best speed/quality balance |
| 384 text-to-image | 8 | 107.8s | 71.7s | Highest validated 384 quality |
| 512 text-to-image | 4 | 100.2s | 63.1s | Best speed/quality balance |
| 512 text-to-image | 8 | 159.7s | 123.0s | Highest validated 512 run |
| 512 no-PyTorch text-to-image | 4 | 117.4s | 80.1s | TRT VAE + TRT text encoder |
| 512 no-PyTorch img2img | 8, strength 0.65 | 129.7s | 91.6s | 5 effective denoise steps |
| 384 img2img | 8, strength 0.65 | 83.9s | 46.0s | 5 effective denoise steps |
| 384 no-PyTorch img2img | 8, strength 0.65 | 123.1s | 86.9s | Single-stage VAE encode + denoise |

Cache limits on Orin NX 16GB:

| Resolution | Default cache | OOM boundary |
|---|---:|---|
| 384 | 23 layers | 24 layers OOM during step 1 |
| 512 | 18 layers | 19 layers OOM during step 1 |

## Quickstart

This is the fastest path when you want to run the API on a Jetson. The
no-PyTorch runtime uses TensorRT engines for inference, so it only needs a few
small config/tokenizer files from the upstream model, not the full PyTorch
weights.

1. Download the minimal runtime config from the upstream model:

```bash
hf download Tongyi-MAI/Z-Image-Turbo \
  --local-dir "$HOME/models/z-image-turbo-fp8-diffusers" \
  --include "model_index.json" \
  --include "tokenizer/*" \
  --include "scheduler/scheduler_config.json" \
  --include "vae/config.json"
```

2. Download the prebuilt TensorRT artifacts:

```bash
hf download harvestsu/z-image-turbo-jetson-trt-artifacts \
  --local-dir "$HOME/models/z-image-trt-artifacts"
```

3. Pull the small runtime image:

```bash
docker pull sensecraft-missionpack.seeed.cn/solution/z-image-jetson-no-torch:latest
```

4. Start the 512px HTTP API:

```bash
DOCKER_IMAGE=sensecraft-missionpack.seeed.cn/solution/z-image-jetson-no-torch:latest \
RESOLUTION=512 \
MAX_CACHED_LAYERS=18 \
MODEL_ROOT_HOST=$HOME/models \
ENGINE_DIR_512_HOST=$HOME/models/z-image-trt-artifacts/engines/orin-nx-jp6-trt10.3/512-bf16 \
TEXT_ENCODER_ENGINE_DIR_HOST=$HOME/models/z-image-trt-artifacts/engines/orin-nx-jp6-trt10.3/text-encoder-split-g4 \
OUTPUT_DIR_HOST=$HOME/z-image-output \
UPLOAD_DIR_HOST=$HOME/z-image-input/api-uploads \
API_PORT=8000 \
scripts/run/run_3drope_no_torch_api.sh
```

5. Generate an image:

```bash
curl -X POST http://<jetson-ip>:8000/generate \
  -F 'prompt=A cute orange tabby cat sitting on a sunny windowsill, photorealistic' \
  -F 'num_steps=4' \
  -F 'output_name=cat_512.png'
```

The output is saved on the Jetson host under `$HOME/z-image-output` and served
from `http://<jetson-ip>:8000/outputs/cat_512.png`.

### Artifact Layout

This repo does not commit model weights, ONNX files, or TensorRT engines to
normal git. The runtime image also does not contain those files. The expected
deployment layout is:

```text
$HOME/models/
  z-image-turbo-fp8-diffusers/              # minimal config/tokenizer files
  z-image-trt-artifacts/                    # HF artifact download
    engines/orin-nx-jp6-trt10.3/
      384-bf16/
      512-bf16/
      text-encoder-split-g4/
```

You can reproduce the artifacts from the public upstream model, or download the
published TensorRT artifacts from Hugging Face:

```bash
hf download harvestsu/z-image-turbo-jetson-trt-artifacts \
  --local-dir "$HOME/models/z-image-trt-artifacts"
```

See [docs/ARTIFACTS.md](docs/ARTIFACTS.md) for what belongs in GitHub, what
belongs in Hugging Face artifacts, and what stays local.

### Script Runtime

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
ENGINE_DIR_384_HOST=/path/to/z-image-trt-artifacts/engines/orin-nx-jp6-trt10.3/384-bf16 \
ENGINE_DIR_512_HOST=/path/to/z-image-trt-artifacts/engines/orin-nx-jp6-trt10.3/512-bf16 \
TEXT_ENCODER_ENGINE_DIR_HOST=/path/to/z-image-trt-artifacts/engines/orin-nx-jp6-trt10.3/text-encoder-split-g4 \
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

PyTorch-buffer runtime:

```bash
RESOLUTION=384 \
NUM_STEPS=8 \
STRENGTH=0.65 \
INPUT_IMAGE_PATH=/path/to/reference.png \
OUTPUT_PATH=/output/output_img2img.png \
PROMPT="A cute orange tabby cat wearing a small red scarf, photorealistic" \
scripts/run/run_3drope_basic_refiner.sh
```

No-PyTorch runtime:

```bash
DOCKER_IMAGE=sensecraft-missionpack.seeed.cn/solution/z-image-jetson-no-torch:latest \
RESOLUTION=384 \
NUM_STEPS=8 \
STRENGTH=0.65 \
MAX_CACHED_LAYERS=18 \
INPUT_IMAGE_PATH=/path/to/reference.png \
OUTPUT_PATH=/output/output_img2img_no_torch.png \
PROMPT="A cute orange tabby cat wearing a small red scarf, photorealistic" \
scripts/run/run_3drope_no_torch.sh
```

The no-PyTorch img2img path runs VAE encode and denoise in one container by
default. Set `IMG2IMG_TWO_STAGE=1` to force a two-process fallback that writes
the init latent to `/output/init_latent_no_torch.npz` before denoising.
Set `RESOLUTION=512` and point `ENGINE_DIR_512_HOST` at the 512 engine folder
to run the validated 512 path.

`STRENGTH` controls how much the reference image changes:

- `0.3-0.5`: preserve composition strongly, small edits
- `0.6-0.7`: good prompt/edit balance
- `0.8+`: larger semantic changes, more drift

## HTTP API

The no-PyTorch runtime image can also run as a small HTTP service. The API runs
one request at a time and launches the TensorRT pipeline in a child process per
request so CUDA/TensorRT memory is released after each generation.

Start the service:

```bash
DOCKER_IMAGE=sensecraft-missionpack.seeed.cn/solution/z-image-jetson-no-torch:latest \
RESOLUTION=512 \
MAX_CACHED_LAYERS=18 \
MODEL_ROOT_HOST=$HOME/models \
ENGINE_DIR_512_HOST=$HOME/models/z-image-trt-artifacts/engines/orin-nx-jp6-trt10.3/512-bf16 \
TEXT_ENCODER_ENGINE_DIR_HOST=$HOME/models/z-image-trt-artifacts/engines/orin-nx-jp6-trt10.3/text-encoder-split-g4 \
OUTPUT_DIR_HOST=$HOME/z-image-output \
UPLOAD_DIR_HOST=$HOME/z-image-input/api-uploads \
API_PORT=8000 \
scripts/run/run_3drope_no_torch_api.sh
```

`MODEL_ROOT_HOST`, `ENGINE_DIR_384_HOST`, `ENGINE_DIR_512_HOST`,
`TEXT_ENCODER_ENGINE_DIR_HOST`, `OUTPUT_DIR_HOST`, and `UPLOAD_DIR_HOST` are
host paths and can be changed to match your deployment. The launcher mounts
`OUTPUT_DIR_HOST` into the container as `/output` and `UPLOAD_DIR_HOST` as
`/uploads`.

Keep the service at one uvicorn worker. The API uses an in-process lock as a
simple FIFO queue: concurrent requests wait, and only one TensorRT generation
runs at a time. Running multiple workers or multiple containers on one Jetson
can start multiple generations and OOM the device.

Health check:

```bash
curl http://<jetson-ip>:8000/health
```

Text-to-image:

```bash
curl -X POST http://<jetson-ip>:8000/generate \
  -F 'prompt=A cute orange tabby cat sitting on a sunny windowsill, photorealistic' \
  -F 'num_steps=4' \
  -F 'output_name=cat_512.png'
```

Img2img with an uploaded reference image:

```bash
curl -X POST http://<jetson-ip>:8000/generate \
  -F 'prompt=A cute orange tabby cat wearing a small red scarf, photorealistic' \
  -F 'image=@/path/to/reference.png' \
  -F 'num_steps=8' \
  -F 'strength=0.65' \
  -F 'output_name=cat_scarf_512.png'
```

JSON call when the reference image is already visible inside the container:

```bash
curl -X POST http://<jetson-ip>:8000/generate_json \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A cute orange tabby cat wearing a small red scarf, photorealistic",
    "image_path": "/uploads/reference.png",
    "num_steps": 8,
    "strength": 0.65,
    "output_name": "cat_scarf_512.png"
  }'
```

Successful response:

```json
{
  "success": true,
  "mode": "img2img",
  "resolution": 512,
  "num_steps": 8,
  "strength": 0.65,
  "seed": 42,
  "elapsed_seconds": 129.7,
  "trt_seconds": 91.6,
  "image_path": "/output/cat_scarf_512.png",
  "image_url": "/outputs/cat_scarf_512.png"
}
```

The output is both saved on the host and served over HTTP. In the example above:

```text
Host file: $OUTPUT_DIR_HOST/cat_scarf_512.png
HTTP URL:  http://<jetson-ip>:8000/outputs/cat_scarf_512.png
```

Download the generated image:

```bash
curl http://<jetson-ip>:8000/outputs/cat_scarf_512.png -o cat_scarf_512.png
```

Failures return `success: false` with an `error` string.

Validated API smoke tests on Orin NX 16GB:

| Endpoint | Mode | Settings | Result |
|---|---|---|---|
| `/generate_json` | text2img | 512, 4 steps | `success:true`, 118.648s total, 78.3s TRT |
| `/generate_json` | img2img | 512, 4 steps, strength 0.65 | `success:true`, 86.276s total, 46.4s TRT |

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

For a visual adaptation story with real success and failure images, see
[docs/ARTICLE_Z_IMAGE_TENSORRT_JETSON.md](docs/ARTICLE_Z_IMAGE_TENSORRT_JETSON.md).

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
  for the faster Python runtime, or the published no-PyTorch runtime image for
  the slim path

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

Published no-PyTorch runtime image:

```bash
docker pull sensecraft-missionpack.seeed.cn/solution/z-image-jetson-no-torch:latest

DOCKER_IMAGE=sensecraft-missionpack.seeed.cn/solution/z-image-jetson-no-torch:latest \
RESOLUTION=384 \
scripts/run/run_3drope_no_torch.sh
```

The pushed image digest is:

```text
sha256:9cbcb5a2df638f70f4cfc60c68f7ed6f88fc4984bba9491efc451415787eadeb
```

Build the no-PyTorch runtime locally if you need to modify the image:

```bash
docker build \
  -f docker/Dockerfile.runtime-jetson-no-torch \
  -t z-image-jetson-no-torch:latest .
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
- `scripts/run/pipeline_trt_no_torch.py` is an experimental text-to-image
  runtime that avoids importing PyTorch and uses TensorRT + CUDA Runtime via
  `ctypes`. It is validated for 384 text-to-image on Orin NX 16GB; by default
  it caches all 30 transformer layer engines in 384 mode.
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
