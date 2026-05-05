---
license: apache-2.0
tags:
  - z-image
  - tensorrt
  - jetson
  - orin-nx
  - image-generation
  - text-to-image
  - edge-ai
library_name: tensorrt
base_model: Tongyi-MAI/Z-Image-Turbo
---

# Z-Image-Turbo Jetson TensorRT Artifacts

This repository hosts generated artifacts for running
[Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
with TensorRT on NVIDIA Jetson.

Code, Dockerfiles, export scripts, and reproduction docs live in the companion
source repository. If you publish the source under a different GitHub
organization/name, update this section to point there.

## Validated Target

Current published artifacts are intended for this target unless a folder name
states otherwise:

| Field | Value |
|---|---|
| Device | Jetson Orin NX 16GB |
| JetPack | 6.x |
| OS | Ubuntu 22.04 |
| CUDA | 12.6 host libraries |
| TensorRT | 10.3 |
| Precision | BF16 TensorRT engines |
| Runtime | TensorRT Python bindings |

TensorRT engines are not portable across arbitrary GPUs, TensorRT versions, or
JetPack/CUDA stacks. If your target differs, rebuild engines from ONNX on that
target.

## Layout

```text
runtime-minimal/
  tokenizer/
  scheduler/
  vae/config.json

engines/
  orin-nx-jp6-trt10.3/
    384-bf16/
      *.engine
    512-bf16/
      *.engine
    text-encoder-split-g4/
      *.engine

manifests/
  engines-orin-nx-jp6-trt10.3-384-bf16.sha256
  engines-orin-nx-jp6-trt10.3-512-bf16.sha256
  text-encoder-split-g4-orin-nx-jp6-trt10.3.sha256
```

`runtime-minimal/` contains only small config/tokenizer files required by the
no-PyTorch runtime. It does not contain the full upstream model weights.

## Expected Size

Approximate validated artifact sizes:

| Artifact | Size |
|---|---:|
| 384 BF16 TensorRT engines | ~12GB |
| 512 BF16 TensorRT engines | ~12GB |
| Split text encoder engines, group size 4 | ~15GB |
| Minimal tokenizer/scheduler/VAE config | Small |

The original upstream diffusers snapshot is about 20GB and is not duplicated
here. Download it from `Tongyi-MAI/Z-Image-Turbo` when using export or PyTorch
fallback paths.

## Use From Runtime

Example host layout after downloading this artifact repo:

```text
/home/harvest/models/z-image-trt-artifacts/
  engines/orin-nx-jp6-trt10.3/384-bf16/
  engines/orin-nx-jp6-trt10.3/text-encoder-split-g4/
  runtime-minimal/
```

Run the no-PyTorch 384 text-to-image path:

```bash
DOCKER_IMAGE=z-image-jetson-no-torch:latest \
MODEL_ROOT_HOST=/home/harvest/models/z-image-trt-artifacts \
ENGINE_DIR_384_HOST=/home/harvest/models/z-image-trt-artifacts/engines/orin-nx-jp6-trt10.3/384-bf16 \
TEXT_ENCODER_ENGINE_DIR_HOST=/home/harvest/models/z-image-trt-artifacts/engines/orin-nx-jp6-trt10.3/text-encoder-split-g4 \
MODEL_DIR=/models/runtime-minimal \
RESOLUTION=384 \
scripts/run/run_3drope_no_torch.sh
```

## Validation Reference

Validated on Jetson Orin NX 16GB:

| Runtime | Resolution | Steps | Total | TRT denoise | Result |
|---|---:|---:|---:|---:|---|
| PyTorch-buffer TRT | 384 | 4 | 73.2s | 37.1s | Correct |
| PyTorch-buffer TRT | 512 | 4 | 100.2s | 63.1s | Correct |
| no-PyTorch TRT | 384 | 4 | 92.8s | 56.2s | Correct |

The no-PyTorch runtime image is 413MB and imports TensorRT, CUDA Runtime through
`ctypes`, NumPy, Pillow, and `tokenizers`. It does not import PyTorch,
diffusers, or transformers.

## License

The source project code is MIT licensed.

These artifacts are derived from `Tongyi-MAI/Z-Image-Turbo`, whose model
license is Apache-2.0. Follow the upstream model terms when redistributing or
using generated artifacts.
