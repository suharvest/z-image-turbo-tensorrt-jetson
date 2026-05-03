# Z-Image-Turbo TensorRT Handoff Document

## Current State (2026-05-02)

### What Works
- **FP8 → BF16 inference** on Jetson Orin NX 16GB via PyTorch + diffusers
- 512×512, 8 steps: **~3 min** (~22s/step), output quality verified (84K colors)
- CUDA sm_87 supported via NVIDIA Jetson torch 2.5.0 (L4T build)
- Docker iptables fixed: `DOCKER_INSECURE_NO_IPTABLES_RAW=1` systemd override

### Docker Images on orin-nx
| Image | Size | Contents |
|-------|------|----------|
| `z-image-turbo:latest` | 7.64GB | ubuntu:22.04 + torch 2.6.0 (broken sm_87) + diffusers |
| `z-image-jetson:latest` | 11.6GB | Same base + NVIDIA L4T torch 2.5.0 + numpy 1.26.4 |

### Model Files on orin-nx
| Path | Size | Description |
|------|------|-------------|
| `~/models/z-image-turbo/` | 31GB | Original fp16 diffusers model (Tongyi-MAI) |
| `~/models/z-image-turbo-fp8/` | 6.2GB | FP8 source (drbaph, ComfyUI format) |
| `~/models/z-image-turbo-fp8-diffusers/` | 12.3GB | Converted FP8→BF16 diffusers format (WORKING) |

### Key Scripts (local: `~/project/image_gen/scripts/`)
| File | Purpose |
|------|---------|
| `gen_image.py` | Main inference script (BF16, 512×512, 8 steps) |
| `convert_fp8_stream.py` | ComfyUI→diffusers format converter (FP8→BF16) |
| `export_onnx.py` | ONNX export attempt (WIP, crashes on model load) |
| `patch_diffusers.py` | Jetson monkey-patches (device_mesh, NP_SUPPORTED_MODULES, GQA) |
| `Dockerfile.zimage` | Docker build file |
| `docker-compose.yml` | Docker Compose config |

### Generated Images
| File | Resolution | Colors | Notes |
|------|-----------|--------|-------|
| `output_bf16.png` | 256×256 | 47K | First success (1 step, blurry) |
| `output_512.png` | 512×512 | 84K | 8 steps, good quality, 2m59s |

---

## TensorRT Implementation Plan

### Phase 1: ONNX Export (1-2 days)
**Goal**: Export ZImageTransformer2DModel to ONNX

**Steps:**
1. Fix container with writable output mount for ONNX
2. Test `torch.onnx.export()` on transformer module
3. Handle likely ONNX export failures:
   - `torch.onnx.export` may fail with unsupported ops (AdaLN, RoPE, dynamic shapes)
   - Fallback: use `torch.export` + `torch.onnx.dynamo_export` (PyTorch 2.1+)
   - Fallback: use `onnxscript` for manual graph construction
4. Verify ONNX model correctness (compare outputs with PyTorch)
5. ONNX opset: target opset 18+ for best TRT compatibility

**Key files to create:**
- `scripts/export_transformer_onnx.py` — working ONNX export
- `scripts/verify_onnx.py` — compare ONNX vs PyTorch outputs

### Phase 2: TRT Engine Build (2-3 days)
**Goal**: Build TensorRT engine from ONNX on Jetson

**Steps:**
1. Mount host TensorRT 10.3 libs into container (or use host directly)
2. Install `tensorrt` Python package matching host version (10.3.0.30)
3. Build engine via Python API or `trtexec`:
   ```bash
   trtexec --onnx=transformer_512.onnx \
     --fp16 --dynamicShapes=... \
     --saveEngine=transformer_512.engine
   ```
4. Handle common TRT build failures:
   - Unsupported ONNX ops → write TRT plugins
   - AdaLN modulation → custom plugin (~200 lines CUDA)
   - RoPE → implement as pre-attention plugin
5. Test engine: load and run inference, compare with PyTorch

**Key files to create:**
- `scripts/build_trt_engine.py` — Python TRT engine builder
- `scripts/test_trt_engine.py` — load engine and compare outputs
- `scripts/trt_plugins/` — custom TRT plugins if needed

### Phase 3: Full Pipeline Integration (2-3 days)
**Goal**: Replace PyTorch transformer with TRT engine in inference loop

**Steps:**
1. Create TRT wrapper class:
   ```python
   class TRTZImageTransformer:
       def __init__(self, engine_path):
           # Load engine, create execution context
       def __call__(self, hidden_states, timestep, encoder_hidden_states, ...):
           # Run TRT inference
   ```
2. Integrate with existing ZImagePipeline (replace transformer)
3. Handle dynamic shapes (different resolutions)
4. Memory management (TRT engine + PyTorch tensors sharing GPU memory)

**Key files to create:**
- `scripts/trt_transformer.py` — TRT wrapper compatible with diffusers pipeline
- `scripts/gen_trt.py` — end-to-end TRT inference script

### Phase 4: Optimization (1-2 weeks)
**Goal**: Maximize speed and minimize memory

**Steps:**
1. INT8 calibration (if quality acceptable)
2. CUDA graph capture for diffusion loop
3. Multi-stream execution (overlap text encoding with denoising)
4. Batch text encoder (if generating multiple images)
5. Profile with `nsys` to identify remaining bottlenecks

---

## Known Issues & Gotchas

### 1. Container vs Host: Two Worlds
- **Container** has working torch + diffusers but NO TensorRT
- **Host** has TensorRT 10.3 but broken torch (numpy + cusparseLt issues)
- **Solution**: Export ONNX in container, build TRT engine on host, run in container

### 2. Fleet Exec Timeouts
- `fleet exec` with sleep >30s often returns exit 143
- `fleet exec` with pip install or model loading returns "Error:" (actual command started)
- **Workaround**: Write wrapper scripts, use `nohup ... &`, check output file later
- Or SSH directly: `ssh orinnx`

### 3. Read-Only Model Mounts
- Container mounts `~/models:/models:ro`
- Need separate writable mount for ONNX/TRT outputs
- Use `~/models-trtexec:/output` or write to `/tmp` inside container

### 4. TensorRT Python Bindings
- Host has TRT 10.3 Python package at `/usr/lib/python3/dist-packages/tensorrt/`
- Need to mount or install matching version in container
- TRT 10.x uses new API (`tensorrt.Runtime`, `tensorrt.Builder`)

### 5. DiT-Specific TRT Challenges
- **AdaLN**: Not standard LayerNorm — scale/shift from time embedding
- **RoPE**: TRT doesn't natively support rotary position encoding
- **Dual Attention**: Self + Cross attention in one pass
- **Dynamic shapes**: Different resolutions need different engine configs

### 6. NP_SUPPORTED_MODULES Missing
- Jetson torch 2.5.0 missing `torch._dynamo.utils.NP_SUPPORTED_MODULES`
- Needed for `torch.onnx.dynamo_export` (the newer ONNX exporter)
- Monkey-patch: `du.NP_SUPPORTED_MODULES = {}`

---

## Quick Reference Commands

### Generate Image (current working method)
```bash
# On orin-nx
docker run --rm --privileged --network=host \
  -v /usr/lib/aarch64-linux-gnu:/host-libs:ro \
  -v /etc/alternatives:/etc/alternatives:ro \
  -v /usr/local/cuda-12.6:/usr/local/cuda:ro \
  -v ~/.local/lib/python3.10/site-packages/nvidia:/usr/local/nvidia-pip:ro \
  -e LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:/usr/local/cuda/nvvm/lib64:/host-libs:/host-libs/tegra:/host-libs/openblas-pthread:/usr/local/nvidia-pip/cusparselt/lib" \
  -v ~/models:/models:ro \
  -v ~/z-image-output:/output \
  -v /tmp/gen_image.py:/workspace/gen_image.py:ro \
  --shm-size=4g \
  z-image-jetson:latest python3 -u /workspace/gen_image.py
```

### Convert FP8 Weights
```bash
docker run --rm --privileged --network=host \
  -v /tmp/convert_fp8_stream.py:/tmp/convert.py:ro \
  -v ~/models:/models \
  z-image-turbo:latest python3 /tmp/convert.py
```

### Start Interactive Container
```bash
docker run -it --rm --privileged --network=host \
  -v /usr/lib/aarch64-linux-gnu:/host-libs:ro \
  -v /etc/alternatives:/etc/alternatives:ro \
  -v /usr/local/cuda-12.6:/usr/local/cuda:ro \
  -v ~/.local/lib/python3.10/site-packages/nvidia:/usr/local/nvidia-pip:ro \
  -e LD_LIBRARY_PATH="..." \
  -v ~/models:/models:ro \
  --shm-size=4g \
  z-image-jetson:latest bash
```

### Check Device Status
```bash
fleet status          # Check all devices
fleet exec orin-nx -- 'free -h && df -h /'
fleet exec orin-nx -- 'docker ps -a'
```

---

## Community References
- [CSDN: Z-Image-ComfyUI TensorRT](https://blog.csdn.net/weixin_31749299/article/details/157424243) — 1.88× on RTX 4090
- [AXERA ONNX→NPU Z-Image-Turbo](https://huggingface.co/AXERA-TECH/Z-Image-Turbo) — Complete ONNX pipeline for AX650N
- [NVIDIA TRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) — LLM TRT framework (no DiT)
- [Gotchas: Docker iptables fix](~/project/app_collaboration/.claude/skills/solution-validation/references/gotchas-jetson.md)

## Next Session Actions
1. Fix host torch or container TRT bindings
2. Run `export_onnx.py` successfully → produce transformer ONNX
3. Build TRT engine from ONNX (on host or container)
4. Benchmark PyTorch vs TRT inference speed
5. If TRT works: integrate into gen_image.py pipeline
6. If TRT fails with unsupported ops: assess plugin effort vs alternative approaches
