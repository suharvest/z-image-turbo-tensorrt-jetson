# Artifacts

This project has several artifact classes. The public Hugging Face artifact
repo created for this project is:

```text
https://huggingface.co/harvestsu/z-image-turbo-jetson-trt-artifacts
```

| Artifact | Typical size | Should it be in git? | Recommended distribution |
|---|---:|---|---|
| Model weights | ~20GB | No | Upstream Hugging Face model |
| ONNX exports | ~12GB per resolution | No | Optional Hugging Face artifact repo |
| TensorRT engines | ~12GB per resolution | No | Hugging Face artifact repo, separated by Jetson/TRT target |
| Split text encoder engines | ~15GB | No | Hugging Face artifact repo, separated by Jetson/TRT target |
| Minimal runtime configs | Small | No | Hugging Face artifact repo |

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

## Public artifact layout

Use the separate Hugging Face model repository:

```text
harvestsu/z-image-turbo-jetson-trt-artifacts
  runtime-minimal/
    tokenizer/
    scheduler/
    vae/config.json
  engines/
    orin-nx-jp6-trt10.3/
      384-bf16/
      512-bf16/
      text-encoder-split-g4/
  manifests/
    engines-orin-nx-jp6-trt10.3-384-bf16.sha256
    engines-orin-nx-jp6-trt10.3-512-bf16.sha256
    text-encoder-split-g4-orin-nx-jp6-trt10.3.sha256
    runtime-minimal.sha256
```

The `text-encoder-split-g4/` folder also contains the tiny
`bf16_to_fp16_1x128x2560.engine` cast engine used by the no-PyTorch runtime to
connect BF16 text encoder output to the FP16 prompt preprocessor without a CPU
round trip.

Keep TensorRT engines separated by hardware/software target. Engines are not a
portable model format; they are tied to TensorRT version, GPU architecture, and
often JetPack/CUDA details.

The checked-in template under `artifacts/hf/` is the initial Hugging Face repo
card and manifest directory. Upload it with:

```bash
hf upload harvestsu/z-image-turbo-jetson-trt-artifacts artifacts/hf . \
  --repo-type model \
  --commit-message "Add artifact repository card"
```

Generate a SHA256 manifest before uploading a large folder:

```bash
scripts/artifacts/make_sha256_manifest.sh \
  /path/to/trt-engines-384-bf16 \
  artifacts/hf/manifests/engines-orin-nx-jp6-trt10.3-384-bf16.sha256
```

Then upload the folder to its target path:

```bash
hf upload harvestsu/z-image-turbo-jetson-trt-artifacts \
  /path/to/trt-engines-384-bf16 \
  engines/orin-nx-jp6-trt10.3/384-bf16 \
  --repo-type model \
  --commit-message "Add Orin NX 384 BF16 engines"
```

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
