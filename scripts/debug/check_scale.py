"""Compare PT vs TRT for each component - memory-optimized for Jetson Orin NX."""
import torch, math, gc, os, time
import torch.distributed as dist
if not hasattr(dist, "device_mesh"):
    dist.device_mesh = type("_", (), {"DeviceMesh": type("_", (), {})})
import torch._dynamo.utils as du
if not hasattr(du, "NP_SUPPORTED_MODULES"):
    du.NP_SUPPORTED_MODULES = {}
import torch.nn.functional as F
_orig_sdpa = F.scaled_dot_product_attention
F.scaled_dot_product_attention = lambda *a, **kw: _orig_sdpa(*a, **{k:v for k,v in kw.items() if k!="enable_gqa"})
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
import tensorrt as trt

ED = "/engines-v2"

class TRTE:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.ins, self.oshapes = [], {}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            sh = tuple(self.engine.get_tensor_shape(n))
            m = self.engine.get_tensor_mode(n)
            if m == trt.TensorIOMode.INPUT:
                self.ins.append(n)
            else:
                self.oshapes[n] = sh

    def __call__(self, **kw):
        for n in self.ins:
            t = kw[n].contiguous().cuda()
            self.ctx.set_tensor_address(n, t.data_ptr())
            self.ctx.set_input_shape(n, tuple(t.shape))
        out = {}
        for n, sh in self.oshapes.items():
            ss = [1 if v == -1 else v for v in list(sh)]
            t = torch.empty(tuple(ss), dtype=torch.float16, device="cuda")
            self.ctx.set_tensor_address(n, t.data_ptr())
            out[n] = t
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return out


# Phase 1: Load full pipeline with memory limits (matching pipeline_trt_v2.py)
print("Phase 1: Loading ZImagePipeline with device_map=balanced, max_memory=12GB...", flush=True)
from diffusers import ZImagePipeline
pipe = ZImagePipeline.from_pretrained(
    "/models/z-image-turbo-fp8-diffusers",
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    max_memory={0: "12GB"},
)
print("Pipeline loaded!", flush=True)
t = pipe.transformer
tok = pipe.tokenizer

# Phase 2: Get prompt_embeds from text encoder
print("Phase 2: Encoding prompt...", flush=True)
ti = tok("A cat", padding="max_length", max_length=128, truncation=True, return_tensors="pt")
with torch.no_grad():
    pe = pipe.text_encoder(
        input_ids=ti.input_ids.to("cuda"),
        attention_mask=ti.attention_mask.to("cuda"),
    ).last_hidden_state
print(f"  prompt_embeds: {pe.shape} [{pe.min():.4f}, {pe.max():.4f}]", flush=True)

# Free text_encoder to make room for TRT engines
del pipe.text_encoder
gc.collect()
torch.cuda.empty_cache()
print("  Text encoder freed", flush=True)

# Phase 3: Compute all PT reference outputs
print("Phase 3: Computing PT reference outputs...", flush=True)

# 3a. cap_embedder (prompt_preprocessor)
with torch.no_grad():
    pt_pp = t.cap_embedder(pe.to(torch.bfloat16))
print(f"  PT cap_embedder ok: {pt_pp.shape} [{pt_pp.min():.4f}, {pt_pp.max():.4f}]", flush=True)

# 3b. x_embedder (latent_preprocessor)
# MUST match ONNX export patch order: (C, H_tok, pH, W_tok, pW) → permute(0,2,4,3,5,1) → (H_tok*W_tok, pH*pW*C)
torch.manual_seed(42)
latent = torch.randn(1, 16, 64, 64, device="cuda")
B, C, H, W = latent.shape
pH = pW = 2
hT, wT = H // pH, W // pW
patches = latent.view(B, C, hT, pH, wT, pW).permute(0, 2, 4, 3, 5, 1).reshape(B, hT * wT, pH * pW * C)
with torch.no_grad():
    pt_lp = t.all_x_embedder["2-1"](patches.to(torch.bfloat16))
print(f"  PT x_embedder ok: {pt_lp.shape} [{pt_lp.min():.4f}, {pt_lp.max():.4f}]", flush=True)

# 3c. t_embedder
t_val = torch.tensor([500.0])
te_mod = t.t_embedder
half = te_mod.frequency_embedding_size // 2
freqs_te = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half)
args = t_val[:, None].float() * freqs_te[None]
t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
with torch.no_grad():
    pt_te_out = te_mod.mlp(t_freq.to(torch.bfloat16))
print(f"  PT t_embedder ok: {pt_te_out.shape} [{pt_te_out.min():.4f}, {pt_te_out.max():.4f}]", flush=True)

# 3d. Build RoPE freqs_cis as COMPLEX (cos + i*sin)
# Z-Image uses COMPLEX freqs_cis format for apply_rotary_emb
axes_dims = [32, 48, 48]
axes_lens = [1536, 512, 512]
seq = 1152
rope_dim = sum(axes_dims)  # 128

# Generate freqs for each axis
all_freqs = []
for dim, length in zip(axes_dims, axes_lens):
    # freqs shape: [length, dim/2]
    freq = 1.0 / (256.0 ** (torch.arange(0, dim, 2).float() / dim))
    t_idx = torch.arange(length).float()
    angles = torch.outer(t_idx, freq)  # [length, dim/2]
    all_freqs.append(angles)  # list of [length, dim/2]

# Build pose_ids for each token
pose_ids = torch.zeros(seq, 3, dtype=torch.long)
for i in range(1024):
    pose_ids[i, 0] = i // 32    # height
    pose_ids[i, 1] = i % 32     # width
    # pose_ids[i, 2] = 0 already (time)

# Gather angles per token
gathered_angles = []
for ax in range(3):
    idx = pose_ids[:, ax]  # [seq]
    gathered_angles.append(all_freqs[ax][idx])  # [seq, dim/2]

# Concatenate: [seq, 16+24+24] = [seq, 64]
angles = torch.cat(gathered_angles, dim=-1)  # [1152, 64]

# Convert to complex: freqs_cis = e^(i*angle) = cos(angle) + i*sin(angle)
freqs_cis = torch.polar(torch.ones_like(angles), angles)  # complex [1152, 64]
freqs_cis = freqs_cis.unsqueeze(0)  # [1, 1152, 64]

attn = torch.ones(1, seq, dtype=torch.bool)
print(f"  RoPE ready: complex {freqs_cis.shape}", flush=True)

# 3e. layer_0
x_pt = torch.cat([pt_lp.to(torch.bfloat16), pt_pp.to(torch.bfloat16)], dim=1)
with torch.no_grad():
    l0_pt = t.layers[0](
        x_pt,
        attn_mask=attn,
        freqs_cis=freqs_cis,
        adaln_input=pt_te_out.to(torch.bfloat16),
    )
print(f"  PT layer_0 ok: {l0_pt.shape} [{l0_pt.min():.4f}, {l0_pt.max():.4f}]", flush=True)

# 3f. final_projection
fl = t.all_final_layer["2-1"]
with torch.no_grad():
    s = 1.0 + fl.adaLN_modulation(pt_te_out.to(torch.bfloat16)).unsqueeze(1)
    n = fl.norm_final(l0_pt.to(torch.bfloat16)) * s
    n = fl.linear(n)
    n = n[:, :1024, :]
    B_pt = n.shape[0]
    outC = 16
    Ht = Wt = 32
    pt_npred = n.view(B_pt, Ht, Wt, pH, pW, outC).permute(0, 5, 1, 3, 2, 4).reshape(B_pt, outC, Ht * pH, Wt * pW)
print(f"  PT noise_pred ok: {pt_npred.shape} [{pt_npred.min():.4f}, {pt_npred.max():.4f}]", flush=True)

# Free transformer (no longer needed - TRT engines replace it)
del pipe, t
gc.collect()
torch.cuda.empty_cache()
print("  Transformer freed", flush=True)

# Phase 4: Load TRT engines and compare
print("Phase 4: Loading TRT engines...", flush=True)
pp_e = TRTE(f"{ED}/prompt_preprocessor_fp16.engine")
lp_e = TRTE(f"{ED}/latent_preprocessor_fp16.engine")
te_e = TRTE(f"{ED}/t_embedder_fp16.engine")
l0_e = TRTE(f"{ED}/layer_00_fp16.engine")
fin_e = TRTE(f"{ED}/final_projection_fp16.engine")
print("  All 5 engines loaded", flush=True)

# For TRT layer_0, we need freqs_cis as real interleaved (FP16) [1, 1152, 128]
# The TRT engine expects this format
freqs_cis_real = torch.view_as_real(freqs_cis).flatten(2)  # [1, 1152, 128] real interleaved
print(f"  TRT freqs_cis: {freqs_cis_real.shape}", flush=True)

print("\n" + "=" * 70, flush=True)
print("COMPONENT-BY-COMPONENT PT vs TRT COMPARISON", flush=True)
print("=" * 70, flush=True)

# 1. prompt_preprocessor
trt_pp = pp_e(prompt_embeds=pe.to(torch.float16))["processed_prompt"]
d = (trt_pp.float() - pt_pp.float().cuda()).abs()
print("\n--- 1. prompt_preprocessor (cap_embedder) ---", flush=True)
print(f"  PT:  [{pt_pp.min():.6f}, {pt_pp.max():.6f}] shape={pt_pp.shape}", flush=True)
print(f"  TRT: [{trt_pp.min():.6f}, {trt_pp.max():.6f}] shape={trt_pp.shape}", flush=True)
print(f"  abs_diff: mean={d.mean():.6f} max={d.max():.6f}", flush=True)
pp_ratio = (abs(trt_pp.max()) + abs(trt_pp.min()) + 1e-10) / (abs(pt_pp.max()) + abs(pt_pp.min()) + 1e-10)
print(f"  SCALE RATIO: {pp_ratio:.4f}x", flush=True)

# 2. latent_preprocessor
trt_lp = lp_e(latent=latent.to(torch.float32))["image_tokens"]
d = (trt_lp.float() - pt_lp.float().cuda()).abs()
print("\n--- 2. latent_preprocessor (x_embedder) ---", flush=True)
print(f"  PT:  [{pt_lp.min():.6f}, {pt_lp.max():.6f}]", flush=True)
print(f"  TRT: [{trt_lp.min():.6f}, {trt_lp.max():.6f}]", flush=True)
print(f"  abs_diff: mean={d.mean():.6f} max={d.max():.6f}", flush=True)
lp_ratio = (abs(trt_lp.max()) + abs(trt_lp.min()) + 1e-10) / (abs(pt_lp.max()) + abs(pt_lp.min()) + 1e-10)
print(f"  SCALE RATIO: {lp_ratio:.4f}x", flush=True)

# 3. t_embedder
trt_te_out = te_e(timestep=t_val.to(torch.float32))["adaln_input"]
d = (trt_te_out.float() - pt_te_out.float().cuda()).abs()
print("\n--- 3. t_embedder ---", flush=True)
print(f"  PT:  [{pt_te_out.min():.6f}, {pt_te_out.max():.6f}]", flush=True)
print(f"  TRT: [{trt_te_out.min():.6f}, {trt_te_out.max():.6f}]", flush=True)
print(f"  abs_diff: mean={d.mean():.6f} max={d.max():.6f}", flush=True)
te_ratio = (abs(trt_te_out.max()) + abs(trt_te_out.min()) + 1e-10) / (abs(pt_te_out.max()) + abs(pt_te_out.min()) + 1e-10)
print(f"  SCALE RATIO: {te_ratio:.4f}x", flush=True)

# 4. layer_0
x_trt = torch.cat([trt_lp, trt_pp], dim=1)
l0_trt = l0_e(
    x=x_trt,
    attn_mask=attn.to("cuda"),
    freqs_cis=freqs_cis_real.to(torch.float16).to("cuda"),
    adaln_input=trt_te_out,
)["output"]
d = (l0_trt.float() - l0_pt.float().cuda()).abs()
print("\n--- 4. layer_0 ---", flush=True)
print(f"  PT:  [{l0_pt.min():.4f}, {l0_pt.max():.4f}]", flush=True)
print(f"  TRT: [{l0_trt.min():.4f}, {l0_trt.max():.4f}]", flush=True)
print(f"  abs_diff: mean={d.mean():.6f} max={d.max():.6f}", flush=True)
l0_ratio = (abs(l0_trt.max()) + abs(l0_trt.min()) + 1e-10) / (abs(l0_pt.max()) + abs(l0_pt.min()) + 1e-10)
print(f"  SCALE RATIO: {l0_ratio:.4f}x", flush=True)

# 5. final_projection
trt_npred = fin_e(hidden=l0_trt, adaln_input=trt_te_out)["noise_pred"]
d = (trt_npred.float() - pt_npred.float().cuda()).abs()
print("\n--- 5. final_projection ---", flush=True)
print(f"  PT:  [{pt_npred.min():.4f}, {pt_npred.max():.4f}] shape={pt_npred.shape}", flush=True)
print(f"  TRT: [{trt_npred.min():.4f}, {trt_npred.max():.4f}] shape={trt_npred.shape}", flush=True)
print(f"  abs_diff: mean={d.mean():.6f} max={d.max():.6f}", flush=True)
npred_ratio = (abs(trt_npred.max()) + abs(trt_npred.min()) + 1e-10) / (abs(pt_npred.max()) + abs(pt_npred.min()) + 1e-10)
print(f"  SCALE RATIO: {npred_ratio:.4f}x", flush=True)

# SUMMARY TABLE
print("\n" + "=" * 70, flush=True)
print("SUMMARY TABLE", flush=True)
print("=" * 70, flush=True)
print(f"{'Component':28s} {'PT_range':24s} {'TRT_range':24s} {'Scale':>8s}  {'diff_mean':>10s}  {'cosine_sim'}", flush=True)
print("-" * 70, flush=True)

cmp = [
    ("1.prompt_preprocessor", pt_pp, trt_pp),
    ("2.latent_preprocessor", pt_lp, trt_lp),
    ("3.t_embedder", pt_te_out, trt_te_out),
    ("4.layer_0", l0_pt, l0_trt),
    ("5.final_projection", pt_npred, trt_npred),
]

for name, pt_t, trt_t in cmp:
    pt_range = f"[{pt_t.min():.2f},{pt_t.max():.2f}]"
    trt_range = f"[{trt_t.min():.2f},{trt_t.max():.2f}]"
    pt_amp = (abs(pt_t.max()) + abs(pt_t.min())) / 2
    trt_amp = (abs(trt_t.max()) + abs(trt_t.min())) / 2
    ratio = max(pt_amp, trt_amp) / (min(pt_amp, trt_amp) + 1e-10)
    d = (trt_t.float().cuda() - pt_t.float().cuda()).abs()
    cos = F.cosine_similarity(
        pt_t.float().cuda().flatten(),
        trt_t.float().cuda().flatten(),
        dim=0,
    )
    print(f"{name:28s} {pt_range:24s} {trt_range:24s} {ratio:7.2f}x  {d.mean():10.4f}  {cos:10.6f}", flush=True)

print("\nDONE", flush=True)
