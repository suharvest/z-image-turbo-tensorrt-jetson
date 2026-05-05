# Reproduction Checklist

Use this checklist when reproducing the project on a new machine.

## 1. Prepare the target Jetson

Validated target:

- Jetson Orin NX 16GB
- JetPack 6 / Ubuntu 22.04
- TensorRT 10.3
- CUDA 12.6 host libraries

Check TensorRT:

```bash
/usr/src/tensorrt/bin/trtexec --version
```

Check Docker:

```bash
docker run --rm hello-world
```

If Docker is not available to the current user, use sudo or add the user to the
docker group according to your system policy.

## 2. Prepare the runtime image

The runtime image only contains dependencies. It does not contain model weights
or engines.

For the validated no-PyTorch 384 text-to-image path, pull the published
MissionPack image:

```bash
docker pull sensecraft-missionpack.seeed.cn/solution/z-image-jetson-no-torch:latest
export DOCKER_IMAGE=sensecraft-missionpack.seeed.cn/solution/z-image-jetson-no-torch:latest
```

Published digest:

```text
sha256:a4c1733edd84e3e87d4fe08fa72514d91628f763874c5f2f15b35cfeae20da1a
```

Either use an existing Jetson PyTorch image:

```bash
export DOCKER_IMAGE=<your-image>
```

Or build the template image:

```bash
docker build \
  --build-arg TORCH_WHEEL_URL=https://example.com/path/to/jetson-torch.whl \
  -f docker/Dockerfile.jetson \
  -t z-image-jetson:latest .
export DOCKER_IMAGE=z-image-jetson:latest
```

## 3. Export ONNX

Run on a CUDA workstation with enough memory:

```bash
MODEL_PATH=Tongyi-MAI/Z-Image-Turbo \
TRANSFORMER_SUBFOLDER=transformer \
RESOLUTION=384 \
OUTPUT_DIR=/path/to/onnx-384 \
python3 scripts/export/export_all_layers_fp16.py

MODEL_PATH=Tongyi-MAI/Z-Image-Turbo \
TRANSFORMER_SUBFOLDER=transformer \
RESOLUTION=384 \
OUTPUT_DIR=/path/to/onnx-384 \
FORCE_EXPORT=1 \
python3 scripts/export/export_refiners.py
```

Repeat with `RESOLUTION=512` and a separate `OUTPUT_DIR` if you need 512 mode.

Optional: export VAE encoder/decoder ONNX to avoid loading PyTorch VAE during
runtime:

```bash
MODEL_PATH=Tongyi-MAI/Z-Image-Turbo \
RESOLUTION=384 \
OUTPUT_DIR=/path/to/onnx-384 \
python3 scripts/export/export_vae.py
```

Optional: export the Qwen3 text encoder ONNX to avoid loading the PyTorch text
encoder during runtime. On Jetson, use the split exporter; the monolithic
36-layer text encoder ONNX can parse but OOMs the TensorRT builder on Orin NX.

```bash
MODEL_PATH=Tongyi-MAI/Z-Image-Turbo \
OUTPUT_DIR=/path/to/onnx-text-encoder-split-g4 \
GROUP_SIZE=4 \
python3 scripts/export/export_text_encoder_split.py
```

## 4. Build TensorRT engines on Jetson

Copy ONNX files to the Jetson, then run:

```bash
ONNX_DIR=/path/to/onnx-384 \
ENGINE_DIR=/path/to/trt-engines-384-bf16 \
scripts/export/build_trt_engines.sh
```

The build script intentionally uses `--bf16`. FP16 is not the validated path for
the transformer or VAE decoder on this pipeline. VAE ONNX exports are named
`*_fp16.onnx` because the exported tensors are FP16, but `vae_decoder_fp16.onnx`
produced NaNs on Orin NX when built with `trtexec --fp16`.

For split text encoder engines, build each group explicitly:

```bash
for onnx in /path/to/onnx-text-encoder-split-g4/*.onnx; do
  base="$(basename "$onnx" .onnx)"
  /usr/src/tensorrt/bin/trtexec \
    --onnx="$onnx" \
    --saveEngine="/path/to/trt-text-encoder-split-g4/${base}.engine" \
    --bf16 \
    --builderOptimizationLevel=0 \
    --memPoolSize=workspace:1024 \
    --skipInference
done
```

## 5. Run text-to-image

```bash
DOCKER_IMAGE=z-image-jetson:latest \
MODEL_ROOT_HOST=/path/to/models \
MODEL_DIR=/models/z-image-turbo-fp8-diffusers \
ENGINE_DIR_384_HOST=/path/to/trt-engines-384-bf16 \
OUTPUT_DIR_HOST=/path/to/output \
CUDA_HOST=/usr/local/cuda-12.6 \
TRT_PY_HOST=/usr/lib/python3.10/dist-packages/tensorrt \
NVIDIA_PIP_HOST=/path/to/python/site-packages/nvidia \
RESOLUTION=384 \
scripts/run/run_3drope_basic_refiner.sh
```

If VAE engines were built in the selected engine directory, enable the TRT VAE
path:

```bash
USE_TRT_VAE=1 \
RESOLUTION=384 \
scripts/run/run_3drope_basic_refiner.sh
```

If split text encoder engines were built, enable the TRT text encoder path:

```bash
USE_TRT_TEXT_ENCODER=1 \
TEXT_ENCODER_ENGINE_DIR=/models/axera-onnx/trt-text-encoder-split-g4 \
TEXT_ENCODER_GROUPS=0-3,4-7,8-11,12-15,16-19,20-23,24-27,28-31,32-35 \
USE_TRT_VAE=1 \
RESOLUTION=384 \
scripts/run/run_3drope_basic_refiner.sh
```

This validated TRT runtime path uses the lightweight `tokenizers` loader and the
built-in FlowMatch scheduler by default. Set `USE_TRANSFORMERS_TOKENIZER=1` or
`USE_DIFFUSERS_SCHEDULER=1` only for parity comparisons.

Expected validated reference on Orin NX 16GB:

- 384, 4 steps: about 73 seconds total
- 384, 4 steps with split TRT text encoder + TRT VAE: about 101 seconds total
- 384, 4 steps with experimental no-PyTorch text-to-image runtime: about 93 seconds total
- 384, 8 steps / strength 0.65 with no-PyTorch img2img runtime: about 123 seconds total
- 512, 4 steps: about 100 seconds total
- 512, 4 steps with no-PyTorch text-to-image runtime: about 117 seconds total
- 512, 8 steps / strength 0.65 with no-PyTorch img2img runtime: about 130 seconds total

Optional no-PyTorch text-to-image runtime:

```bash
DOCKER_IMAGE=sensecraft-missionpack.seeed.cn/solution/z-image-jetson-no-torch:latest \
MODEL_ROOT_HOST=/path/to/models \
ENGINE_DIR_384_HOST=/path/to/trt-engines-384-bf16 \
TEXT_ENCODER_ENGINE_DIR_HOST=/path/to/trt-text-encoder-split-g4 \
OUTPUT_DIR_HOST=/path/to/output \
RESOLUTION=384 \
scripts/run/run_3drope_no_torch.sh
```

If you need to modify the runtime image, build it locally with
`docker/Dockerfile.runtime-jetson-no-torch` and set
`DOCKER_IMAGE=z-image-jetson-no-torch:latest`.

## 6. Run img2img

PyTorch-buffer runtime:

```bash
DOCKER_IMAGE=z-image-jetson:latest \
MODEL_ROOT_HOST=/path/to/models \
MODEL_DIR=/models/z-image-turbo-fp8-diffusers \
ENGINE_DIR_384_HOST=/path/to/trt-engines-384-bf16 \
OUTPUT_DIR_HOST=/path/to/output \
CUDA_HOST=/usr/local/cuda-12.6 \
TRT_PY_HOST=/usr/lib/python3.10/dist-packages/tensorrt \
NVIDIA_PIP_HOST=/path/to/python/site-packages/nvidia \
INPUT_IMAGE_PATH=/path/to/reference.png \
RESOLUTION=384 \
NUM_STEPS=8 \
STRENGTH=0.65 \
PROMPT="A cute orange tabby cat wearing a small red scarf, photorealistic" \
scripts/run/run_3drope_basic_refiner.sh
```

No-PyTorch runtime:

```bash
DOCKER_IMAGE=sensecraft-missionpack.seeed.cn/solution/z-image-jetson-no-torch:latest \
MODEL_ROOT_HOST=/path/to/models \
ENGINE_DIR_384_HOST=/path/to/trt-engines-384-bf16 \
TEXT_ENCODER_ENGINE_DIR_HOST=/path/to/trt-text-encoder-split-g4 \
OUTPUT_DIR_HOST=/path/to/output \
INPUT_IMAGE_PATH=/path/to/reference.png \
RESOLUTION=384 \
NUM_STEPS=8 \
STRENGTH=0.65 \
MAX_CACHED_LAYERS=18 \
PROMPT="A cute orange tabby cat wearing a small red scarf, photorealistic" \
scripts/run/run_3drope_no_torch.sh
```

The no-PyTorch img2img path runs VAE encode and denoise in one container by
default. If you hit OOM on a tighter target, set `IMG2IMG_TWO_STAGE=1` to force
a two-process fallback that writes `/output/init_latent_no_torch.npz` before
denoising.

For 512 img2img, set `RESOLUTION=512` and provide `ENGINE_DIR_512_HOST` with
the 512 TensorRT engines, including `vae_encoder_fp16.engine` and
`vae_decoder_fp16.engine`.

## 7. Report your result

When reporting performance, include:

- Jetson model and memory size
- JetPack, CUDA, TensorRT versions
- Resolution
- Step count
- `MAX_CACHED_LAYERS`
- Total time and TRT denoise time
- Whether text-to-image or img2img
