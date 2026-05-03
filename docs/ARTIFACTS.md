# Artifacts

This project has three artifact classes:

| Artifact | Typical size | Should it be in git? | Recommended distribution |
|---|---:|---|---|
| Model weights | ~30GB+ | No | Upstream Hugging Face model |
| ONNX exports | ~12GB per resolution | No | Hugging Face dataset/model repo or GitHub Release with Git LFS |
| TensorRT engines | ~12-13GB per resolution | No | Per-device release assets, because engines are TensorRT/GPU/JetPack dependent |

## Why ONNX is not committed directly

The ONNX export for one resolution contains 38 files: 30 transformer layers,
pre/post processors, context refiners, and noise refiners. A single 512 or 384
set is roughly 12GB. Committing those files to normal git would make clone,
diff, and pull operations impractical.

The recommended pattern is:

1. Keep source scripts and manifests in this repository.
2. Publish large generated artifacts separately.
3. Link the exact artifact release from `README.md`.

## Reproducible export

The export scripts default to the public upstream model:

```bash
MODEL_PATH=Tongyi-MAI/Z-Image-Turbo
TRANSFORMER_SUBFOLDER=transformer
```

Export 384 ONNX:

```bash
RESOLUTION=384 \
OUTPUT_DIR=/path/to/onnx-384 \
python3 scripts/export/export_all_layers_fp16.py

RESOLUTION=384 \
OUTPUT_DIR=/path/to/onnx-384 \
FORCE_EXPORT=1 \
python3 scripts/export/export_refiners.py
```

Export 512 ONNX:

```bash
RESOLUTION=512 \
OUTPUT_DIR=/path/to/onnx-512 \
python3 scripts/export/export_all_layers_fp16.py

RESOLUTION=512 \
OUTPUT_DIR=/path/to/onnx-512 \
FORCE_EXPORT=1 \
python3 scripts/export/export_refiners.py
```

If you already have a local model snapshot, point `MODEL_PATH` at the repo root
and keep `TRANSFORMER_SUBFOLDER=transformer`. If `MODEL_PATH` points directly at
the transformer folder, set `TRANSFORMER_SUBFOLDER=`.

## Recommended public artifact layout

Use a separate Hugging Face repository such as:

```text
<org>/z-image-turbo-jetson-trt-artifacts
  onnx-384/
  onnx-512/
  engines/orin-nx-jp6-trt10.3/384-bf16/
  engines/orin-nx-jp6-trt10.3/512-bf16/
  manifests/
    onnx-384.sha256
    onnx-512.sha256
    engines-384.sha256
    engines-512.sha256
```

Keep TensorRT engines separated by hardware/software target. Engines are not a
portable model format; they are tied to TensorRT version, GPU architecture, and
often JetPack/CUDA details.

## If you still want ONNX in this git repo

Use Git LFS and do it intentionally:

```bash
git lfs install
git lfs track "*.onnx"
git lfs track "*.engine"
git add .gitattributes
```

This is not the default here because the repository is easier to clone and audit
when source code and large generated artifacts are separated.
