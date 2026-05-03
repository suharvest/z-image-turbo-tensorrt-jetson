# Z-Image-Turbo on Jetson Orin NX 16GB

## Ranked approaches

### 1. FP8 transformer + staged T5 text encoding

- Peak unified memory: about 11-14 GB during denoising after T5 is released; about 8-10 GB during prompt encoding.
- Estimated time: 3-8 minutes for 1024x1024 at 8-9 steps, depending on clocks, attention backend, and memory pressure.
- Trade-offs: requires converting the ComfyUI checkpoint to Diffusers layout while preserving FP8 tensors. Prompt encoding is a separate stage, so batching should stay at 1. This is the most practical route because it avoids the 12.3 GB FP16 transformer and avoids keeping T5-XXL resident during denoising.

### 2. FP8 transformer + CPU/GPU offload for the full pipeline

- Peak unified memory: about 13-16 GB.
- Estimated time: 5-12 minutes for 1024x1024 at 8-9 steps.
- Trade-offs: easier to wire through `ZImagePipeline.from_pretrained`, but Jetson unified memory means CPU offload does not create a separate memory pool. It mainly reduces CUDA allocator pressure, not total RAM pressure. This can still OOM when page cache or desktop services are active.

### 3. FP16 transformer + staged T5 with aggressive resolution fallback

- Peak unified memory: about 14.5-16+ GB at 768x768, usually too high at 1024x1024.
- Estimated time: 4-10 minutes at 512-768 px.
- Trade-offs: uses the already converted 12.3 GB transformer, but leaves very little room for activations, VAE, allocator fragmentation, and temporary tensors. It is useful only as a fallback for lower resolutions.

### 4. Quantize T5-XXL to 8-bit while keeping FP8 transformer

- Peak unified memory: about 9-13 GB.
- Estimated time: 3-8 minutes plus some overhead for quantized text encoding.
- Trade-offs: good if `bitsandbytes` is already working on the Jetson container, but it was not listed as guaranteed. The provided runnable path uses only torch, diffusers, accelerate, transformers, numpy, and PIL.

## Top recommendation

Use an FP8-preserving Diffusers conversion for `drbaph/Z-Image-Turbo-FP8`, then run inference in two stages:

1. Load only tokenizer + T5 text encoder, encode the prompt, move prompt embeddings to CPU, and fully release T5.
2. Load `ZImagePipeline` without the text encoder/tokenizer, keeping the transformer in native `torch.float8_e4m3fn`, then denoise from the precomputed prompt embeddings.

This is the best fit for Jetson Orin NX 16GB because the transformer is about 6.2 GB in FP8 instead of 12.3 GB in FP16, and the 7.5 GB T5-XXL encoder is not resident at the same time as the denoiser. The scripts below also print memory snapshots and fall back to smaller dimensions on generation OOM.

## Docker run command

Adjust host model/output paths if yours differ:

```bash
docker run --rm -it --privileged --runtime nvidia --network host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8 \
  -e CUDA_MODULE_LOADING=LAZY \
  -e TOKENIZERS_PARALLELISM=false \
  -e HF_HUB_DISABLE_TELEMETRY=1 \
  -v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra:ro \
  -v /usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/local/cuda-12.6/targets/aarch64-linux/lib:ro \
  -v /Users/harvest/project/image_gen:/workspace \
  -v /path/on/jetson/models:/models \
  -v /path/on/jetson/output:/output \
  -w /workspace \
  z-image-turbo:latest \
  python3 /workspace/run_zimage_optimized.py \
    --model /models/z-image-turbo-fp8-diffusers \
    --base_model /models/z-image-turbo \
    --prompt "A cinematic photo of a small robot on a workbench" \
    --steps 8 \
    --guidance_scale 0 \
    --output /output/zimage.png
```

First-time conversion inside the same container:

```bash
python3 /workspace/convert_model.py \
  --src_base /models/z-image-turbo \
  --src_fp8 /models/z-image-turbo-fp8/z_image_turbo_fp8_e4m3fn.safetensors \
  --dst /models/z-image-turbo-fp8-diffusers
```
