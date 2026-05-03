"""Batch download (wget) and build TRT engines for AXERA ONNX files on orin-nx."""
import os, sys, time, subprocess

ONNX_DIR = "/home/harvest/models/axera-onnx/transformer_onnx"
ENGINE_DIR = "/home/harvest/models/axera-onnx/trt-engines"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
MIRROR_BASE = "https://hf-mirror.com/AXERA-TECH/Z-Image-Turbo/resolve/main"

os.makedirs(ENGINE_DIR, exist_ok=True)

# All files we need (from list_repo_files)
ALL_FILES = [
    "cfg_00_timestep_to_model_t_embedder_mlp_mlp_2_Gemm_output_0_config.onnx",
    "cfg_01_prompt_embeds_to_model_Slice_1_output_0_config.onnx",
    "cfg_02_latent_model_input_to_model_Slice_output_0_config.onnx",
] + [
    f"cfg_{n:02d}_model_layers_{n-3}_Add_4_output_0_to_model_layers_{n-2}_Add_4_output_0_config.onnx"
    for n in range(3, 33)
] + [
    "auto_00_model_layers_29_Add_4_output_0_to_sample_auto.onnx",
]

def run_cmd(cmd, timeout=900):
    return os.system(cmd)

for i, fname in enumerate(ALL_FILES):
    onnx_path = os.path.join(ONNX_DIR, fname)
    engine_path = os.path.join(ENGINE_DIR, fname + ".engine")
    log_path = os.path.join(ENGINE_DIR, fname + ".log")

    if os.path.exists(engine_path):
        size_mb = os.path.getsize(engine_path) / 1e6
        print(f"[{i+1}/{len(ALL_FILES)}] SKIP {fname[:50]}... engine exists ({size_mb:.0f}MB)", flush=True)
        # Still clean up ONNX if present
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        continue

    # Download if needed
    if not os.path.exists(onnx_path):
        url = f"{MIRROR_BASE}/transformer_onnx/{fname}"
        print(f"[{i+1}/{len(ALL_FILES)}] DL {fname[:60]}...", flush=True)
        ret = os.system(f"wget -q --show-progress '{url}' -O '{onnx_path}' 2>&1")
        if ret != 0 or not os.path.exists(onnx_path):
            print(f"  DOWNLOAD FAILED (ret={ret})", flush=True)
            continue
        size_mb = os.path.getsize(onnx_path) / 1e6
        print(f"  Downloaded: {size_mb:.0f}MB", flush=True)

    # Build TRT engine
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"[{i+1}/{len(ALL_FILES)}] BUILD {fname[:50]}... ({size_mb:.0f}MB)", flush=True)
    t0 = time.time()
    cmd = f"{TRTEXEC} --onnx={onnx_path} --saveEngine={engine_path} --fp16 > {log_path} 2>&1"
    ret = os.system(cmd)
    elapsed = time.time() - t0

    if ret == 0 and os.path.exists(engine_path):
        eng_mb = os.path.getsize(engine_path) / 1e6
        print(f"  OK: {size_mb:.0f}MB -> {eng_mb:.0f}MB engine in {elapsed:.0f}s", flush=True)
        os.remove(onnx_path)  # Save space
        print(f"  Deleted ONNX source", flush=True)
    else:
        print(f"  BUILD FAILED (ret={ret}) after {elapsed:.0f}s", flush=True)
        # Keep ONNX for debugging

    # Report disk
    stat = os.statvfs("/")
    free_gb = (stat.f_frsize * stat.f_bavail) / 1e9
    print(f"  Disk free: {free_gb:.1f}GB", flush=True)

print("ALL DONE", flush=True)
