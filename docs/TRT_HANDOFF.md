# Z-Image-Turbo TensorRT Handoff Document

> Historical engineering log. For a fresh reproduction path, start with
> [README.md](../README.md) and [REPRODUCTION.md](REPRODUCTION.md). Paths in
> this file may refer to the original validation machines.

## TL;DR

Z-Image-Turbo (6B DiT) on Jetson Orin NX via TensorRT BF16: split-engine pipeline now runs end-to-end and generates correct cat images at 512x512 and 384x384. PyTorch baseline on wsl2 remains available for ONNX export and quality comparison.

**Status**: Fixed basic-mode refiner mismatch. TensorRT BF16 pipeline now generates a clean, recognizable cat image on orin-nx.

## 2026-05-03 Update

Root cause of the blurry/multi-exposure output was the `noise_refiner` export/call path. Z-Image basic mode calls `noise_refiner` with `noise_mask=None`, so the block must use global `adaln_input` modulation. The previous TRT export only implemented the per-token `noise_mask/t_noisy/t_clean` branch and the pipeline forced `noise_mask=ones`, which did not match PyTorch forward semantics.

Fixed files:
- `scripts/export/export_refiners.py`: re-exported `noise_refiner_00/01` with inputs `x, attn_mask, freqs_cis, adaln_input`.
- `scripts/run/pipeline_trt_v2.py`: calls basic-mode `noise_refiner` with `adaln_input`, loads context refiners, and allocates TRT outputs by engine dtype.
- orin-nx `/tmp/pipe_3drope.py` and `/tmp/run_3drope.sh`: updated to the fixed pipeline.

Rebuilt engines on orin-nx:
- `/home/harvest/models/axera-onnx/trt-engines-bf16/noise_refiner_00.engine`
- `/home/harvest/models/axera-onnx/trt-engines-bf16/noise_refiner_01.engine`
- `/home/harvest/models/axera-onnx/trt-engines-bf16/context_refiner_00.engine`
- `/home/harvest/models/axera-onnx/trt-engines-bf16/context_refiner_01.engine`

Validated output:
- `/home/harvest/z-image-output/output_trt_v2.png` was generated with TensorRT engines and is a clean photorealistic orange tabby cat on a windowsill.
- `/tmp/run_3drope.sh` now writes future runs to `/home/harvest/z-image-output/output_3drope.png`.

Performance update:
- `MAX_CACHED_LAYERS=18` is the best validated setting on orin-nx 16GB after switching to minimal PyTorch load, delayed VAE load, freeing TRT engines before VAE decode, releasing prompt/context engines before denoise, and reusing two layer output buffers.
- Full 512x512 8-step run: 159.7s wall, 123.0s TRT denoise.
- `MAX_CACHED_LAYERS=19` loads but OOM-kills during step 1. `20/30` OOM before denoise.
- 512 step sweep picked 4 steps as the default speed/quality balance: 100.2s wall, 63.1s TRT denoise. 3 steps is usable but softer; 2 steps is not acceptable. Use 5 steps for a conservative quality bump.
- 384x384 mode added with separate static-shape BF16 engines at `/home/harvest/models/axera-onnx/trt-engines-384-bf16/`.
- Full 384x384 8-step run with `MAX_CACHED_LAYERS=23`: 107.8s wall, 71.7s TRT denoise. Output: `/home/harvest/z-image-output/output_384.png`.
- 384 cache 24 OOMs during step 1, so cache 23 is the default for `RESOLUTION=384`.
- 384 step sweep picked 4 steps as the default speed/quality balance: 73.2s wall, 37.1s TRT denoise. 3 steps is usable but visibly weaker; 2 steps is not acceptable.
- Img2img mode added: `INPUT_IMAGE_PATH=/host/image.png` + `STRENGTH=...` uses VAE encode and `scheduler.scale_noise(...)`, then reuses the existing TRT denoise path. Validated 384 example: `NUM_STEPS=8 STRENGTH=0.65` generated a red-scarf cat from a cat reference in 83.9s wall / 46.0s TRT.

---

## Hardware & Network

| Machine | Specs | IP (Tailscale) | Role |
|---------|-------|----------------|------|
| orin-nx | Jetson Orin NX, 16GB LPDDR5, 238GB NVMe | 100.82.225.102 | TRT inference target |
| wsl2-local | RTX 3060 12.9GB, 32GB RAM, 658GB disk | 100.73.210.80 | ONNX export + PyTorch baseline |
| Mac | Apple Silicon, 64GB | — | Control node |

- All devices on Tailscale mesh
- orin-nx needs HF mirror (`https://hf-mirror.com`); wsl2 uses `http://127.0.0.1:7890`
- **Prefer SSH over fleet exec**: `ssh harvest@100.82.225.102` works directly, avoids fleet timeout issues
- **Prefer fleet exec for wsl2**: `fleet exec wsl2-local -- 'command'`

---

## What's Been Built

### ONNX Export (wsl2 → orin-nx)
- **Script**: wsl2:`/home/harve/trt-work/export_all_layers_fp32_adaln.py`
- 512 ONNX files were exported on wsl2 and built into `/home/harvest/models/axera-onnx/trt-engines-bf16/`
- 384 ONNX files are at wsl2:`/home/harve/trt-work/onnx-384/` and built into `/home/harvest/models/axera-onnx/trt-engines-384-bf16/`
- Each layer: 386MB, 13 FP16 + 2 FP32 (AdaLN) weights, 4 Clip ops, opset 17
- Key fixes in export:
  - **AdaLN FP32**: `tanh()` and `1.0 + scale` done in FP32 before FP16 cast
  - **Clamped FFN**: `torch.clamp(..., ±60000)` at 3 stages
  - **Wrapper matches original forward**: `SingleLayerWrapper.forward` follows `ZImageTransformerBlock.forward`

### TRT Engines (orin-nx)
- **Location**: `/home/harvest/models/axera-onnx/trt-engines-bf16/` (38 engines, 13GB)
- **384 Location**: `/home/harvest/models/axera-onnx/trt-engines-384-bf16/` (38 engines, 12GB)
- **Build command**: `trtexec --onnx=X.onnx --saveEngine=X.engine --bf16 --timingCacheFile=/tmp/trt_cache.txt`
- All 38 engines PASSED (30 layers + 4 pre/post + 4 refiners)
- **Important**: Must use `--bf16` NOT `--fp16`. FP16 causes attention softmax overflow → layers become identity

384 mode reuses shape-compatible engines for `t_embedder`, `prompt_preprocessor`, and `context_refiner_00/01`. It has separate engines for latent/image-token-shape-dependent components: `latent_preprocessor`, `noise_refiner_00/01`, `layer_00..29`, and `final_projection`.

### Pre/Post Processors
| Engine | Input | Output |
|--------|-------|--------|
| prompt_preprocessor | prompt_embeds [1,128,2560] | processed_prompt [1,128,3840] |
| latent_preprocessor | latent [1,16,H/8,W/8] FP32 | image_tokens [1,(H/16)*(W/16),3840] |
| t_embedder | timestep [1] | adaln_input [1,256] |
| final_projection | hidden [1,image_tokens+128,3840] + adaln | noise_pred [1,16,H/8,W/8] |

### Refiner Engines (NEW — from codex analysis)
| Engine | Inputs | Size |
|--------|--------|------|
| noise_refiner_00/01 | x[1,image_tokens,3840], mask[1,image_tokens]BOOL, freqs[1,image_tokens,128], adaln_input[1,256] | 512: 706MB each; 384: 350MB each |
| context_refiner_00/01 | x[1,128,3840], mask[1,128]BOOL, freqs[1,128,128] | 339MB each |

### Layer Engines
- 30 transformer layers, 354MB each
- I/O: x[1,image_tokens+128,3840], attn_mask[1,image_tokens+128]BOOL, freqs_cis[1,image_tokens+128,128], adaln_input[1,256] → output[1,image_tokens+128,3840]

---

## Pipeline Architecture (current)

```
Text Encoder (Qwen3 4B, PyTorch) → prompt_embeds [1,128,2560]
    ↓ hidden_states[-2] (codex fix: use -2 layer output)
TRT prompt_preprocessor → processed_prompt [1,128,3840]
    ↓
TRT context_refiner_00(processed_prompt, cap_mask, cap_freqs_3d) → refined
TRT context_refiner_01(refined, ...) → refined_prompt [1,128,3840]

Random latent [1,16,H/8,W/8]
    ↓
TRT latent_preprocessor → image_tokens [1,(H/16)*(W/16),3840]
    ↓
TRT noise_refiner_00(image_tokens, img_mask, img_freqs_3d, adaln_input) → refined
TRT noise_refiner_01(refined, ...) → refined_img [1,(H/16)*(W/16),3840]

Concat(refined_img, refined_prompt) → x [1,image_tokens+128,3840]

For each of N steps (8 or 20):
    t_embedder(1000 - scheduler_t) → adaln_input [1,256]
    latent_preprocessor(current_latent) → image_tokens
    noise_refiner_00..01(image_tokens, ..., adaln_input=adaln) → refined_img
    Concat → x
    30× TRT layer(x, unified_mask, unified_freqs_3d, adaln) → x
    final_projection(x, adaln) → noise_pred_raw
    noise_pred = -noise_pred_raw.to(float32)  # codex fix: negate
    scheduler.step(noise_pred, t, latent) → latent

VAE Decode:
    latent = (latent / 0.3611) + 0.1159  # VAE scaling
    image = vae.decode(latent)
    image = (image/2 + 0.5).clamp(0,1)
```

### RoPE Configuration (3D)
- axes_dims=[32,48,48], axes_lens=[1536,512,512], theta=256.0
- **Image tokens**: pose_ids `[F=129, H=h_idx, W=w_idx]` for the active token grid, 32x32 at 512 and 24x24 at 384
- **Text tokens**: pose_ids `[1..128, 0, 0]`
- Format: interleaved cos/sin (cos0, sin0, cos1, sin1, ...)
- Separate freqs for image-only, text-only, and unified sequences

---

## Key Fixes Applied (chronological)

1. **AdaLN FP32**: Modulation weights kept in FP32; tanh and 1.0+scale computed in FP32 before FP16 cast
2. **Clamped FFN**: Prevents FP16 overflow in SiLU gating → fixes NaN
3. **3D RoPE**: Correct multi-axis positional encoding matching model config
4. **Token order**: Image first, prompt second
5. **BF16 engines**: FP16 softmax overflow killed attention; BF16 (8-bit exponent) fixes it
6. **Codex pipeline fixes** (applied to `scripts/run/pipeline_trt_v2.py`):
   - timestep = `1000 - scheduler_t` (diffusers convention)
   - noise_pred negated before scheduler
   - FlowMatch scheduler with `mu=calculate_shift(image_tokens)`
   - VAE decode with `scaling_factor` and `shift_factor`
   - RoPE pose_ids matching official format
   - text_encoder `output_hidden_states=True`, use `hidden_states[-2]`
7. **noise_refiner + context_refiner**: 4 additional engines for pre-layer refinement

---

## Current Problem

No current image-quality blocker for the validated cat prompt at 512 or 384. The remaining bottleneck is speed: split static-shape engines are memory-bound on Orin NX, and cache 18 at 512 / cache 23 at 384 are the highest validated settings before OOM.

Historical issue: blurry/multi-exposure output was caused by exporting/calling `noise_refiner` through the per-token `noise_mask/t_noisy/t_clean` branch. Re-exporting basic-mode `noise_refiner` with `adaln_input` fixed it.

---

## Key Files

### On orin-nx (ssh harvest@100.82.225.102)
| Path | Purpose |
|------|---------|
| `/home/harvest/models/axera-onnx/trt-engines-bf16/` | 512 static-shape BF16 TRT engines |
| `/home/harvest/models/axera-onnx/trt-engines-384-bf16/` | 384 static-shape BF16 TRT engines |
| `/home/harvest/models/z-image-turbo-fp8-diffusers/` | BF16 diffusers model |
| `/tmp/pipe_3drope.py` | Current pipeline script |
| `/tmp/run_3drope.sh` | Docker run script for pipeline |
| `/home/harvest/z-image-output/` | Output images |
| `/usr/src/tensorrt/bin/trtexec` | TRT builder |
| `/tmp/trt_cache.txt` | TRT timing cache (use for faster builds) |

### On wsl2-local (fleet exec wsl2-local)
| Path | Purpose |
|------|---------|
| `/home/harve/trt-work/export_all_layers_fp32_adaln.py` | ONNX export script |
| `/home/harve/trt-work/export_refiners.py` | Refiner export script |
| `/home/harve/trt-work/onnx-384/` | Exported 384 ONNX files |
| `/home/harve/pt_cat.png` | PyTorch baseline image (311KB, real cat) |
| `/home/harve/.local/lib/python3.12/site-packages/diffusers/models/transformers/transformer_z_image.py` | Patched diffusers (RoPE real arithmetic) |

### Local (Mac)
| Path | Purpose |
|------|---------|
| `scripts/run/pipeline_trt_v2.py` | Reference pipeline (with codex fixes) |
| `docs/TRT_STATUS.md` | Previous status document |
| `docs/TRT_HANDOFF.md` | This document |

---

## How to Deploy & Test

### 1. Generate an image
```bash
ssh harvest@100.82.225.102 "bash /tmp/run_3drope.sh"
# Default is NUM_STEPS=4 for both 512 and 384. Override with NUM_STEPS=5 or NUM_STEPS=8 for higher quality.
# Output: /home/harvest/z-image-output/output_3drope.png
```

384 mode:
```bash
ssh harvest@100.82.225.102 "RESOLUTION=384 bash /tmp/run_3drope.sh"
# Output: /home/harvest/z-image-output/output_384.png
# Default is NUM_STEPS=4. Override with NUM_STEPS=5 or NUM_STEPS=8 for higher quality.
```

Img2img mode:
```bash
ssh harvest@100.82.225.102 "RESOLUTION=384 NUM_STEPS=8 STRENGTH=0.65 \
  INPUT_IMAGE_PATH=/home/harvest/z-image-input/ref.png \
  OUTPUT_PATH=/output/output_img2img.png \
  PROMPT='A cute orange tabby cat wearing a small red scarf, photorealistic' \
  bash /tmp/run_3drope.sh"
```

`INPUT_IMAGE_PATH` is a host path and is mounted into the container as `/input/init_image`. If the image is already visible inside the container, set `INIT_IMAGE` directly instead. Lower `STRENGTH` preserves the reference more; higher `STRENGTH` gives the prompt more freedom.

### 2. Pull image to Mac
```bash
fleet pull orin-nx /home/harvest/z-image-output/output_3drope.png ~/Downloads/
open ~/Downloads/output_3drope.png
```

### 3. Rebuild engines (if ONNX changes)
```bash
# Copy/export ONNX to a temporary directory on orin-nx first, then build:
ssh harvest@100.82.225.102 "
cd /home/harvest/models/axera-onnx/<onnx-dir>
for f in *.onnx; do
    name=\$(basename \$f .onnx)
    /usr/src/tensorrt/bin/trtexec \
        --onnx=\$f \
        --saveEngine=../<engine-dir>/\${name}.engine \
        --bf16 \
        --timingCacheFile=/tmp/trt_cache.txt
done
"
```

### 4. Test a single engine for identity
```bash
ssh harvest@100.82.225.102 "docker run --rm --privileged --network=host \
  -v /usr/lib/aarch64-linux-gnu:/host-libs:ro \
  -v /usr/local/cuda-12.6:/usr/local/cuda:ro \
  -v /home/harvest/models/axera-onnx/trt-engines-bf16:/engines-bf16:ro \
  -v /home/harvest/.local/lib/python3.10/site-packages/nvidia:/usr/local/nvidia-pip:ro \
  -v /usr/lib/python3.10/dist-packages/tensorrt:/usr/local/trt-py:ro \
  -e PYTHONPATH=/usr/local/trt-py \
  -e LD_LIBRARY_PATH='/usr/local/cuda/lib64:/host-libs:/host-libs/tegra' \
  z-image-jetson:latest python3 -c \"
import torch, tensorrt as trt
class E:
    def __init__(s,p):
        with open(p,'rb') as f:s.e=trt.Runtime(trt.Logger()).deserialize_cuda_engine(f.read())
        s.c=s.e.create_execution_context();s.s=torch.cuda.Stream();s.ins=[];s.os={}
        for i in range(s.e.num_io_tensors):
            n=s.e.get_tensor_name(i);sh=tuple(s.e.get_tensor_shape(n))
            if s.e.get_tensor_mode(n)==trt.TensorIOMode.INPUT:s.ins.append(n)
            else:s.os[n]=sh
    def __call__(s,**kw):
        for n in s.ins:
            t=kw[n].contiguous().cuda();s.c.set_tensor_address(n,t.data_ptr());s.c.set_input_shape(n,tuple(t.shape))
        o={}
        for n,sh in s.os.items():
            ss=[1 if v==-1 else v for v in list(sh)];t=torch.empty(tuple(ss),dtype=torch.float16,device='cuda');s.c.set_tensor_address(n,t.data_ptr());o[n]=t
        s.c.execute_async_v3(s.s.cuda_stream);s.s.synchronize()
        return o

eng=E('/engines-bf16/layer_00_fp16.engine')
x=torch.randn(1,1152,3840,dtype=torch.float16,device='cuda')*7
adaln=torch.randn(1,256,dtype=torch.float16,device='cuda')*0.1
freqs=torch.randn(1,1152,128,dtype=torch.float16,device='cuda')
m=torch.ones(1,1152,dtype=torch.bool,device='cuda')
o=eng(x=x,attn_mask=m,freqs_cis=freqs,adaln_input=adaln)['output']
d=(o-x).float().abs().mean()
print(f'diff={d:.6f} {\"WORKS\" if d>1e-5 else \"DEAD\"}')
\"" 2>&1
```

---

## Next Steps for Agent

1. **wsl2 layer-chain comparison**: Run 30-layer PT-vs-TRT comparison, find first divergent layer
2. **Export fixes**: If divergence found, fix ONNX export for that layer type
3. **Speed optimization**: Cache all 30 layer engines in memory (they fit: 10.6GB for layers + text_encoder freed)
4. **Service packaging**: Wrap pipeline in Flask/FastAPI, add Docker Compose
5. **Context refiner fix**: context_refiner ONNX build FAILED (parse error) — needs re-export with correct wrapper
