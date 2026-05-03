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

## 4. Build TensorRT engines on Jetson

Copy ONNX files to the Jetson, then run:

```bash
ONNX_DIR=/path/to/onnx-384 \
ENGINE_DIR=/path/to/trt-engines-384-bf16 \
scripts/export/build_trt_engines.sh
```

The build script intentionally uses `--bf16`. FP16 is not the validated path for
Z-Image-Turbo on this pipeline.

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

Expected validated reference on Orin NX 16GB:

- 384, 4 steps: about 73 seconds total
- 512, 4 steps: about 100 seconds total

## 6. Run img2img

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

## 7. Report your result

When reporting performance, include:

- Jetson model and memory size
- JetPack, CUDA, TensorRT versions
- Resolution
- Step count
- `MAX_CACHED_LAYERS`
- Total time and TRT denoise time
- Whether text-to-image or img2img
