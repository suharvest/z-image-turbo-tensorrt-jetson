# Z-Image-Turbo TensorRT Acceleration — Status

> Historical engineering log. For a fresh reproduction path, start with
> [README.md](../README.md) and [REPRODUCTION.md](REPRODUCTION.md). Paths in
> this file may refer to the original validation machines.

## Overview
Accelerating Z-Image-Turbo (6B DiT, 30 transformer layers) on NVIDIA Jetson Orin NX (16GB RAM) via TensorRT 10.3.

## Latest Result
As of 2026-05-05, the no-PyTorch runtime path supports both 384x384 text-to-image
and 384x384 img2img on orin-nx from a 413MB runtime image.

Validated no-PyTorch run on orin-nx:
- Local image: `z-image-jetson-no-torch:latest`
- Published image: `sensecraft-missionpack.seeed.cn/solution/z-image-jetson-no-torch:latest`
- Published digest: `sha256:e328d15da5110288dc0341ffa929e983fbd861c6c7ea82c6c185f5d3025542a1`
- Image size: 413MB
- Runtime imports: TensorRT Python, CUDA Runtime through `ctypes`, NumPy, Pillow, tokenizers
- Runtime does not import: PyTorch, diffusers, transformers
- Resolution: 384x384
- Steps: 4
- `MAX_CACHED_LAYERS=30`
- Total wall time: 92.8s
- TRT denoise time: 56.2s
- Per-step times: 29.5s, 8.9s, 9.3s, 8.6s
- Output pulled to local: `/tmp/output_384_no_torch_layercache30.png`
- Output md5: `88c8cf38c83e372d3d1fdb7d2b31a337`

Validated no-PyTorch img2img run on orin-nx:
- Reference image: `/home/harvest/z-image-input/ref_cat_384.png`
- Prompt: `A cute orange tabby cat wearing a small red scarf, photorealistic`
- Resolution: 384x384
- Steps: 8
- Strength: 0.65
- Effective denoise steps: 5
- `MAX_CACHED_LAYERS=18`
- Total wall time: 123.1s
- TRT denoise time: 86.9s
- Per-step times: 30.8s, 14.2s, 13.6s, 13.8s, 14.4s
- Output: `/home/harvest/z-image-output/output_384_no_torch_img2img_single_stage_cache18.png`
- Output md5 after pull: `cb4cd25777cffe5bfbd10cbe48b35403`

Validated 512 no-PyTorch runs on orin-nx:
- Text-to-image, 4 steps, `MAX_CACHED_LAYERS=18`: 117.4s total, 80.1s TRT denoise
  - Per-step times: 33.4s, 15.8s, 15.5s, 15.4s
  - Output: `/home/harvest/z-image-output/output_512_no_torch_text2img_cast.png`
  - Output md5 after pull: `ab054426e1e21903d69f39a68521d9bd`
- Img2img, 8 steps, strength 0.65, `MAX_CACHED_LAYERS=18`: 129.7s total, 91.6s TRT denoise
  - Effective denoise steps: 5
  - Per-step times: 32.2s, 15.5s, 14.5s, 14.4s, 14.9s
  - Output: `/home/harvest/z-image-output/output_512_no_torch_img2img_cast.png`
  - Output md5 after pull: `34b6b00bdcdb11072669ddea8c9e4de1`

The no-PyTorch path caches all 30 denoise layer engines in 384 mode on Orin NX 16GB. This became viable after removing PyTorch/diffusers/transformers runtime residency and reusing two TensorRT layer output buffers. Text encoder split engines are still loaded group-by-group; VAE decoder is loaded at decode time.

For no-PyTorch img2img, the default launcher keeps VAE encode and denoise in one
container. A two-process fallback remains available with `IMG2IMG_TWO_STAGE=1`;
it first writes `/output/init_latent_no_torch.npz`, then starts a fresh process
for denoising and VAE decode. Earlier OOM tests were misleading because
`MAX_CACHED_LAYERS` was not being forwarded into the Docker container.

As of 2026-05-03, the TensorRT BF16 path generates a correct, clean cat image on orin-nx. The remaining major quality bug was not a 30-layer accumulation issue; it was a refiner export/call mismatch:

- PyTorch basic mode calls `noise_refiner` with `noise_mask=None` and uses global `adaln_input`.
- The old TRT `noise_refiner` export only implemented the per-token `noise_mask/t_noisy/t_clean` branch.
- Re-exporting `noise_refiner_00/01` with inputs `x, attn_mask, freqs_cis, adaln_input` and updating the pipeline fixed the multi-exposure/blurry artifact.

Validated outputs on orin-nx:
- 512x512: `/home/harvest/z-image-output/output_3drope.png`
- 384x384: `/home/harvest/z-image-output/output_384.png`

## Performance Sweep
### No-PyTorch Runtime

384x384, 4-step, no-PyTorch runtime-only image on orin-nx:

| Runtime | Image size | Layer cache | Total wall time | TRT denoise | Visual result |
|---|---:|---:|---:|---:|---|
| PyTorch-buffer TRT runtime | 11.7GB | 23 | 101.2s | 36.2s | Correct |
| no-PyTorch runtime | 413MB | sequential load | 114.0s | 74.1s | Correct |
| no-PyTorch runtime | 413MB | 23 | 105.6s | 63.8s | Correct |
| no-PyTorch runtime | 413MB | 30 | 92.8s | 56.2s | Correct |
| no-PyTorch img2img | 413MB | 18 | 123.1s | 86.9s | Correct |
| no-PyTorch 512 text-to-image | 413MB | 18 | 117.4s | 80.1s | Correct |
| no-PyTorch 512 img2img | 413MB | 18 | 129.7s | 91.6s | Correct |

The 30-layer cache is validated only for 384 mode on Orin NX 16GB. 512 mode is
validated with 18 cached layers. 8GB Orin Nano should not assume this memory
budget.

No-PyTorch runtime files:
- `docker/Dockerfile.runtime-jetson-no-torch`
- `scripts/run/pipeline_trt_no_torch.py`
- `scripts/run/run_3drope_no_torch.sh`

### PyTorch-Buffer Runtime

512x512, 8-step, BF16 TRT transformer/refiners on orin-nx:

| Layer engine cache | 2-step TRT time | Status |
|---:|---:|---|
| 5 | 56.4s | Works |
| 10 | 48.7s | Works |
| 15 | 39.5s | Works |
| 16 | 37.5s | Works |
| 17 | — | OOM killed during step 1 |
| 20 | — | OOM killed while loading/starting |

Validated full 8-step run with minimal PyTorch load, delayed VAE, TRT release before decode, prompt/context engine release, layer output buffer reuse, and `MAX_CACHED_LAYERS=18`:
- Total wall time: 159.7s
- TRT denoise time: 123.0s
- Per-step times: 16.9s, 14.9s, 15.0s, 15.4s, 15.6s, 14.8s, 15.2s, 15.3s
- Output: `/home/harvest/z-image-output/output_3drope.png`

`MAX_CACHED_LAYERS=19` loads but OOM-kills during step 1. `MAX_CACHED_LAYERS=20/30` OOM before denoise. Default cache is now 18 in `scripts/run/pipeline_trt_v2.py` and `/tmp/run_3drope.sh`.

512 step sweep with cache 18:

| Steps | Total wall time | TRT denoise | Visual result |
|---:|---:|---:|---|
| 2 | 68.9s | 32.0s | Not acceptable; face/eyes collapse |
| 3 | 84.4s | 47.5s | Usable but softer face/fur |
| 4 | 100.2s | 63.1s | Best speed/quality balance; clean structure and good detail |
| 5 | 114.3s | 77.8s | Slightly better than 4, slower |
| 8 | 159.7s | 123.0s | Highest validated full-run setting, slower |

Default for `RESOLUTION=512` is now `NUM_STEPS=4`. Use `NUM_STEPS=5` for a conservative quality bump or `NUM_STEPS=8` for the highest validated full-run setting.

384x384 split-engine mode uses separate static-shape BF16 engines at `/home/harvest/models/axera-onnx/trt-engines-384-bf16/` and keeps the same Python/TRT pipeline. Default cache is 23 layers.

Validated full 8-step 384 run:
- Total wall time: 107.8s
- TRT denoise time: 71.7s
- Per-step times: 10.5s, 8.7s, 8.8s, 8.6s, 8.7s, 8.9s, 8.7s, 8.8s
- Output: `/home/harvest/z-image-output/output_384.png`

384 step sweep with cache 23:

| Steps | Total wall time | TRT denoise | Visual result |
|---:|---:|---:|---|
| 2 | 55.6s | 19.5s | Not acceptable; face/eyes collapse |
| 3 | 63.8s | 28.1s | Usable but face and fur are visibly weaker |
| 4 | 73.2s | 37.1s | Best speed/quality balance; clean structure and acceptable detail |
| 5 | 82.0s | 46.4s | Slightly better face/fur than 4, slower |
| 6 | 90.4s | 55.0s | Small detail gain over 5 |
| 8 | 107.8s | 71.7s | Best quality, slower |

Default for `RESOLUTION=384` is now `NUM_STEPS=4`. Use `NUM_STEPS=5` for a conservative quality bump or `NUM_STEPS=8` for the highest validated quality.

384 cache sweep:

| Layer engine cache | 2-step TRT time | Status |
|---:|---:|---|
| 18 | 28.1s | Works |
| 20 | 24.0s | Works |
| 22 | 20.8s | Works |
| 23 | 18.7s | Works |
| 24 | — | OOM killed during step 1 |

### 5-layer group engine prototype
Tested `layers_00_04_fp16.engine`:
- ONNX external-data package size: ~1.9GB
- TRT engine size: 1734.88 MiB
- Execution context device memory: 928.966 MiB
- trtexec latency for layers 0-4: ~274ms host, ~273ms GPU compute

Pipeline test with `USE_GROUP_LAYERS=1`:
- `MAX_CACHED_LAYERS=13`: 2-step TRT 36.5s, works
- `MAX_CACHED_LAYERS=14`: 2-step TRT 35.8s, works
- `MAX_CACHED_LAYERS=15`: OOM during step 1

Conclusion: 5-layer group is technically viable but slower than the current split-layer path (`MAX_CACHED_LAYERS=18`, 2-step TRT 31.5s). The group engine's larger context/scratch memory reduces layer cache capacity enough to erase the Python-call savings.

### Python vs C++ TRT call overhead
Benchmarked `layer_00_fp16.engine` with fixed preallocated GPU buffers:

| Runner | Iterations | Average |
|---|---:|---:|
| Python TensorRT binding | 100 | 83.0 ms |
| C++ TensorRT runtime | 100 | 90.8 ms |
| Python TensorRT binding | 300 | 122.3 ms |
| C++ TensorRT runtime | 300 | 132.4 ms |
| trtexec | 10s duration | 166.7 ms GPU compute mean |

Conclusion: C++ invocation is not faster in this minimal benchmark, and Python binding overhead is not the dominant bottleneck. The dominant cost is TensorRT layer execution / memory bandwidth / runtime clock variability, not Python dispatch.

## Test Environments

| Machine | Hardware | OS | Key Software | Role |
|---------|----------|-----|-------------|------|
| orin-nx | Jetson Orin NX, 16GB LPDDR5, 238GB NVMe | Ubuntu 22.04 (JetPack 6) | TRT 10.3, Docker z-image-jetson:latest, Python 3.10 | TRT inference target |
| wsl2-local | RTX 3060 12.9GB, 32GB RAM, 658GB disk | WSL2 Ubuntu 22.04 | PyTorch 2.5.1+cu124, diffusers 0.37.1, Python 3.12 | ONNX export + PyTorch baseline |
| Mac | Apple Silicon, 64GB | macOS 15.4 | zsh, fleet CLI | Main control node |

## Network
- All devices on Tailscale mesh
- orin-nx: 100.82.225.102
- wsl2-local: 100.73.210.80
- orin-nx needs HF mirror (hf-mirror.com) for downloads; wsl2 uses proxy http://127.0.0.1:7890

## What Works

### 1. TRT Engine Build
- 38 ONNX files exported from PyTorch (wsl2→orin-nx) for 512 mode
  - 30 transformer layers (386MB each): 13 FP16 + 2 FP32 (AdaLN) weights, opset 17
  - prompt_preprocessor, latent_preprocessor, t_embedder, final_projection, 2 context refiners, 2 noise refiners
- All TRT engines built with `trtexec --bf16 --timingCacheFile`, all PASSED
- 512 engine location: `/home/harvest/models/axera-onnx/trt-engines-bf16/` on orin-nx
- 384 engine location: `/home/harvest/models/axera-onnx/trt-engines-384-bf16/` on orin-nx

### 2. Key Fixes Applied
- **AdaLN FP32**: Modulation weights (adaLN_modulation.0) kept in FP32 in ONNX; `tanh()` and `1.0 + scale` computed in FP32 before FP16 cast
- **3D RoPE**: Using correct multi-axis positional encoding matching model config `axes_dims=[32,48,48]`, `axes_lens=[1536,512,512]`, `theta=256.0`
- **Token order**: Image tokens first, prompt tokens second
- **Clamped FFN**: `torch.clamp(..., ±60000)` at 3 stages to prevent FP16 overflow in SiLU gating
- **FP16 FFN overflow**: Fixed by clamp (prevents NaN from cuBLAS non-saturating INF)
- **All 30 layers non-NaN**: All layers produce valid non-NaN output for random inputs

### 3. PyTorch Baseline
- wsl2 generates correct cat image: 147s, 512×512, 311KB PNG
- Image confirmed photorealistic by visual inspection
- Model: `Tongyi-MAI/Z-Image-Turbo`, BF16, 8-step Flow Matching

## Current Problems

No current image-quality blocker for the validated cat prompt at 512 or 384. Remaining constraints are feature coverage, performance, and memory:
- no-PyTorch img2img is validated for 384 and 512 on Orin NX 16GB; `IMG2IMG_TWO_STAGE=1` is available as a low-memory fallback.
- no-PyTorch 30-layer cache is validated for 384 on Orin NX 16GB only.
- no-PyTorch text encoder uses a tiny TensorRT `bf16_to_fp16_1x128x2560.engine` cast engine when present, with CPU round trip fallback.
- 512 cache 19 OOMs during step 1; cache 18 is the validated default.
- 384 cache 24 OOMs during step 1; cache 23 is the validated default.
- Orin-nx Docker cannot comfortably keep full PyTorch transformer plus all TRT engines resident, so the pipeline loads minimal PyTorch components, delays VAE load, and releases TRT engines before VAE decode.

Historical image-quality issues now fixed:
- wrong timestep domain into AdaLN (`t` vs `1000-t`)
- missing scheduler sign flip
- missing FlowMatch `mu` shift
- missing VAE scale/shift
- incorrect RoPE pose IDs
- missing context/noise refiners
- basic-mode `noise_refiner` mismatch (`noise_mask` branch vs global `adaln_input`)

## Img2Img Support

Added Z-Image style image-to-image mode to the TRT pipeline:
- `INPUT_IMAGE_PATH=/host/path.png` mounts a host reference image into the container.
- `INIT_IMAGE=/container/path.png` can be used directly if the image is already visible inside the container.
- `STRENGTH` controls how much to transform the reference image. Higher values add more noise and run more denoise steps.
- The implementation uses VAE encode + `scheduler.scale_noise(...)`; the 30 TRT transformer engines are unchanged.

Validated 384 img2img smoke tests:

| Params | Effective steps | Total wall time | TRT denoise | Result |
|---|---:|---:|---:|---|
| `NUM_STEPS=4 STRENGTH=0.45` | 1 | 48.5s | 10.8s | Preserves reference, too little prompt change |
| `NUM_STEPS=8 STRENGTH=0.65` | 5 | 83.9s | 46.0s | Preserves cat/window composition and adds red scarf |

Recommended starting point for 384 img2img: `NUM_STEPS=8 STRENGTH=0.6..0.7`. Use lower strength for reconstruction/style-preserving edits, higher strength for larger semantic edits.

## Pipeline Architecture

```
Text Encoder(Qwen3, PyTorch fallback or TensorRT split engines) → prompt_embeds [1,128,2560]
    ↓
TRT prompt_preprocessor → processed_prompt [1,128,3840]

Random latent [1,16,H/8,W/8]
    ↓
TRT latent_preprocessor → image_tokens [1,(H/16)*(W/16),3840]

Concat(refined image tokens, processed prompt) → x [1,image_tokens+128,3840]

For each of 8 steps:
    TRT t_embedder(timestep) → adaln_input [1,256]
    30× TRT layer(x, attn_mask, freqs_cis, adaln_input) → x
    TRT final_projection(x, adaln_input) → noise_pred [1,16,H/8,W/8]
    FlowMatchEulerDiscreteScheduler.step(noise_pred, t, latent) → new latent

VAE Decoder(PyTorch fallback or TensorRT) → image [H,W,3]
```

## Key Files

| File | Location | Purpose |
|------|----------|---------|
| export_all_layers_fp32_adaln.py | wsl2:/home/harve/trt-work/ | ONNX export with all fixes |
| pipeline_trt_v2.py | local:scripts/ | TRT pipeline (cached loading) |
| pipeline_trt_no_torch.py | local:scripts/run/ | Experimental no-PyTorch TensorRT text-to-image and img2img runtime |
| Dockerfile.runtime-jetson-no-torch | local:docker/ | 413MB no-PyTorch runtime image recipe |
| full_25.py | orin-nx:/tmp/ | No-cache TRT pipeline |
| build_new_engines.sh | orin-nx:/tmp/ | Batch TRT engine build |
| check_scale.py | orin-nx:/tmp/ | PT vs TRT scale comparison |
| TRT engines 512 | orin-nx:/home/harvest/models/axera-onnx/trt-engines-bf16/ | 38 BF16 TRT engines |
| TRT engines 384 | orin-nx:/home/harvest/models/axera-onnx/trt-engines-384-bf16/ | 38 BF16 TRT engines |
| BF16 model | orin-nx:/models/z-image-turbo-fp8-diffusers/ | Diffusers pipeline model |

## Model Config
- Hidden dim: 3840, Heads: 30, KV Heads: 30
- AdaLN dim: 256
- Text encoder: Qwen3, hidden_size=2560, 36 layers
- RoPE: 3D, axes_dims=[32,48,48], axes_lens=[1536,512,512], theta=256.0
- Patch size: 2, Image tokens: 1024 (32×32) at 512 and 576 (24×24) at 384, Text tokens: 128

## Next Steps
1. Keep the patched pipeline semantics:
   - `t_embedder(1000 - scheduler_t)`
   - `noise_pred = -final_projection(...)`
   - `scheduler.set_timesteps(..., mu=calculate_shift(image_tokens))`
   - VAE decode with scaling/shift
   - RoPE pose IDs `[F,H,W]` with caption/image offsets
2. Treat split engines as the default path. The 5-layer group engine was slower after memory pressure reduced the cache budget.
3. For no-PyTorch runtime speed, tune 512 img2img cache/latency and replace the temporary cast engine with a fused final text encoder output if useful.
