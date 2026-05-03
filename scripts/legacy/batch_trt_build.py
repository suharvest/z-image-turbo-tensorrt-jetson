"""Batch download AXERA ONNX files and build TRT engines on orin-nx."""
import os, sys, time, subprocess

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
ONNX_DIR = "/home/harvest/models/axera-onnx/transformer_onnx"
ENGINE_DIR = "/home/harvest/models/axera-onnx/trt-engines"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"

os.makedirs(ENGINE_DIR, exist_ok=True)

FILES = [
    # (huggingface_path, description)
    ("transformer_onnx/auto_00_model_layers_29_Add_4_output_0_to_sample_auto.onnx", "auto_00 final"),
    ("transformer_onnx/cfg_01_prompt_embeds_to_model_Slice_1_output_0_config.onnx", "cfg_01 prompt"),
    ("transformer_onnx/cfg_02_latent_model_input_to_model_Slice_output_0_config.onnx", "cfg_02 latent"),
] + [
    (f"transformer_onnx/cfg_{n:02d}_model_layers_{n-3}_Add_4_output_0_to_model_layers_{n-2}_Add_4_output_0_config.onnx", f"cfg_{n:02d} layer {n-3}->{n-2}")
    for n in range(4, 33)
]

def run(cmd, timeout=600):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return result.stdout + result.stderr

# Check what's already done
existing_onnx = set(os.listdir(ONNX_DIR)) if os.path.isdir(ONNX_DIR) else set()
existing_engines = set(os.listdir(ENGINE_DIR)) if os.path.isdir(ENGINE_DIR) else set()

for hf_path, desc in FILES:
    fname = os.path.basename(hf_path)
    onnx_path = os.path.join(ONNX_DIR, fname)
    engine_path = os.path.join(ENGINE_DIR, fname + ".engine")
    log_path = os.path.join(ENGINE_DIR, fname + ".log")

    if os.path.exists(engine_path):
        size_mb = os.path.getsize(engine_path) / 1e6
        print(f"[SKIP] {desc}: engine exists ({size_mb:.0f}MB)", flush=True)
        continue

    # Download ONNX
    if not os.path.exists(onnx_path):
        print(f"[DL] {desc}...", flush=True)
        from huggingface_hub import hf_hub_download
        try:
            hf_hub_download("AXERA-TECH/Z-Image-Turbo", hf_path, local_dir=os.path.dirname(ONNX_DIR), local_dir_use_symlinks=False)
            print(f"  Downloaded OK", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            continue

    # Build TRT engine
    print(f"[BUILD] {desc}...", flush=True)
    t0 = time.time()
    cmd = f"{TRTEXEC} --onnx={onnx_path} --saveEngine={engine_path} --fp16 > {log_path} 2>&1"
    ret = os.system(cmd)
    elapsed = time.time() - t0

    if ret == 0 and os.path.exists(engine_path):
        size_mb = os.path.getsize(engine_path) / 1e6
        onnx_mb = os.path.getsize(onnx_path) / 1e6
        print(f"  OK: {onnx_mb:.0f}MB ONNX -> {size_mb:.0f}MB engine in {elapsed:.0f}s", flush=True)
        # Delete ONNX to save space
        os.remove(onnx_path)
        print(f"  Deleted ONNX source", flush=True)
    else:
        print(f"  FAILED (ret={ret}) after {elapsed:.0f}s", flush=True)
        print(f"  Check log: {log_path}", flush=True)

    # Report disk
    stat = os.statvfs("/")
    free_gb = (stat.f_frsize * stat.f_bavail) / 1e9
    print(f"  Disk free: {free_gb:.1f}GB", flush=True)

print("DONE - all files processed", flush=True)
