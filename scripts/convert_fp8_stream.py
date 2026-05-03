import torch, os, json, shutil, gc
from safetensors import safe_open
from safetensors.torch import save_file
from collections import OrderedDict

SRC = "/models/z-image-turbo"
FP8 = "/models/z-image-turbo-fp8/z_image_turbo_fp8_e4m3fn.safetensors"
DST = "/models/z-image-turbo-fp8-diffusers"
SHARD_COUNT = 3
OUT_DTYPE = torch.bfloat16

print("Loading FP8 tensors and writing shards...")
os.makedirs(os.path.join(DST, "transformer"), exist_ok=True)

with safe_open(FP8, framework="pt") as f:
    keys = list(f.keys())

def map_key(k):
    if k.startswith("final_layer."):
        return "all_final_layer.2-1." + k[len("final_layer."):]
    if k.startswith("x_embedder."):
        return "all_x_embedder.2-1." + k[len("x_embedder."):]
    if k.startswith("t_embedder."):
        return k  # same name in diffusers
    if "attention.qkv.weight" in k:
        return k.replace("attention.qkv.weight", "__QKV_SPLIT__")
    if "attention.k_norm" in k:
        return k.replace("attention.k_norm", "attention.norm_k")
    if "attention.q_norm" in k:
        return k.replace("attention.q_norm", "attention.norm_q")
    if "attention.out.weight" in k:
        return k.replace("attention.out.weight", "attention.to_out.0.weight")
    return k

shard_size = (len(keys) + SHARD_COUNT - 1) // SHARD_COUNT
shards = [OrderedDict() for _ in range(SHARD_COUNT)]
new_map = {}
shard_idx = 0
tensor_count = 0

with safe_open(FP8, framework="pt") as f:
    for k in keys:
        mk = map_key(k)
        t = f.get_tensor(k)
        if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            t = t.to(OUT_DTYPE)

        if "__QKV_SPLIT__" in mk:
            base = mk.replace("__QKV_SPLIT__", "")
            q, kv, vv = t.chunk(3, dim=0)
            del t
            for sub_t, suffix in [(q, "attention.to_q.weight"),
                                    (kv, "attention.to_k.weight"),
                                    (vv, "attention.to_v.weight")]:
                shards[shard_idx][base + suffix] = sub_t
                new_map[base + suffix] = f"diffusion_pytorch_model-{shard_idx+1:05d}-of-00003.safetensors"
                tensor_count += 1
                if len(shards[shard_idx]) >= shard_size:
                    shard_idx = min(shard_idx + 1, SHARD_COUNT - 1)
        else:
            shards[shard_idx][mk] = t
            new_map[mk] = f"diffusion_pytorch_model-{shard_idx+1:05d}-of-00003.safetensors"
            tensor_count += 1
            if len(shards[shard_idx]) >= shard_size:
                shard_idx = min(shard_idx + 1, SHARD_COUNT - 1)

# Merge all shards into single file to avoid meta tensor loading
merged = OrderedDict()
for shard in shards:
    merged.update(shard)
    del shard
del shards
gc.collect()

print(f"Processed {len(merged)} tensors, writing single safetensors...")
save_file(merged, os.path.join(DST, "transformer", "diffusion_pytorch_model.safetensors"))
print(f"  Written: diffusion_pytorch_model.safetensors ({len(merged)} tensors)")
del merged
gc.collect()

shutil.copy2(os.path.join(SRC, "transformer", "config.json"),
             os.path.join(DST, "transformer", "config.json"))

for comp in ["text_encoder", "vae", "tokenizer", "scheduler"]:
    src = os.path.join(SRC, comp)
    dst = os.path.join(DST, comp)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    print(f"Copied {comp}")

for item in ["model_index.json", "assets"]:
    src = os.path.join(SRC, item)
    dst = os.path.join(DST, item)
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=True)
    else:
        shutil.copy2(src, dst)
    print(f"Copied {item}")

total = sum(os.path.getsize(os.path.join(DST, "transformer", f))
            for f in os.listdir(os.path.join(DST, "transformer"))
            if f.endswith(".safetensors"))
print(f"\nDone! Transformer: {total/1e9:.1f}GB")
