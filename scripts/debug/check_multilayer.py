"""Track PT vs TRT hidden state divergence across 5 layers."""
import torch, math, gc
import torch.distributed as dist
if not hasattr(dist, "device_mesh"):
    dist.device_mesh = type("_", (), {"DeviceMesh": type("_", (), {})})
import torch._dynamo.utils as du
if not hasattr(du, "NP_SUPPORTED_MODULES"):
    du.NP_SUPPORTED_MODULES = {}
import torch.nn.functional as F
_orig = F.scaled_dot_product_attention
F.scaled_dot_product_attention = lambda *a, **kw: _orig(*a, **{k:v for k,v in kw.items() if k!="enable_gqa"})
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
            n = self.engine.get_tensor_name(i); sh = tuple(self.engine.get_tensor_shape(n))
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
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

print("Loading pipeline...", flush=True)
from diffusers import ZImagePipeline
pipe = ZImagePipeline.from_pretrained(
    "/models/z-image-turbo-fp8-diffusers",
    torch_dtype=torch.bfloat16, device_map="balanced", max_memory={0: "12GB"},
)
print("Loaded!", flush=True)
t = pipe.transformer

# Encode prompt
ti = pipe.tokenizer("A cat", padding="max_length", max_length=128, truncation=True, return_tensors="pt")
with torch.no_grad():
    pe = pipe.text_encoder(input_ids=ti.input_ids.cuda(), attention_mask=ti.attention_mask.cuda()).last_hidden_state
del pipe.text_encoder; gc.collect(); torch.cuda.empty_cache()

# PT forward: cap_embedder
with torch.no_grad():
    pt_pp = t.cap_embedder(pe.to(torch.bfloat16))

# PT forward: x_embedder (correct patch order)
torch.manual_seed(42)
latent = torch.randn(1, 16, 64, 64, device="cuda")
B, C, H, W = latent.shape; pH = pW = 2; hT, wT = H//pH, W//pW
patches = latent.view(B, C, hT, pH, wT, pW).permute(0,2,4,3,5,1).reshape(B, hT*wT, pH*pW*C)
with torch.no_grad():
    pt_lp = t.all_x_embedder["2-1"](patches.to(torch.bfloat16))

# PT forward: t_embedder
t_val = torch.tensor([500.0])
half = t.t_embedder.frequency_embedding_size // 2
freqs_t = torch.exp(-math.log(10000)*torch.arange(0,half,dtype=torch.float32)/half)
args_t = t_val[:,None].float()*freqs_t[None]
t_freq = torch.cat([torch.cos(args_t), torch.sin(args_t)], dim=-1)
with torch.no_grad():
    pt_te = t.t_embedder.mlp(t_freq.to(torch.bfloat16))

# RoPE (complex for PT, real-interleaved for TRT)
axes_dims=[32,48,48]; axes_lens=[1536,512,512]; seq=1152
all_f = []
for dim,length in zip(axes_dims,axes_lens):
    fq=1.0/(256.0**(torch.arange(0,dim,2).float()/dim))
    all_f.append(torch.outer(torch.arange(length).float(),fq))
pose = torch.zeros(seq,3,dtype=torch.long)
for i in range(1024): pose[i,0]=i//32; pose[i,1]=i%32
ang = torch.cat([all_f[ax][pose[:,ax]] for ax in range(3)],dim=-1)  # [1152,64]
freqs_cis = torch.polar(torch.ones_like(ang), ang).unsqueeze(0)  # complex [1,1152,64]
freqs_cis_trt = torch.view_as_real(freqs_cis).flatten(2)  # real [1,1152,128]
attn = torch.ones(1, seq, dtype=torch.bool)

# Build x
x_pt = torch.cat([pt_lp.to(torch.bfloat16), pt_pp.to(torch.bfloat16)], dim=1)

# Load TRT engines
pp_e = TRTE(f"{ED}/prompt_preprocessor_fp16.engine")
lp_e = TRTE(f"{ED}/latent_preprocessor_fp16.engine")
te_e = TRTE(f"{ED}/t_embedder_fp16.engine")
layer_trt = [TRTE(f"{ED}/layer_{i:02d}_fp16.engine") for i in range(5)]
fin_e = TRTE(f"{ED}/final_projection_fp16.engine")

trt_pp = pp_e(prompt_embeds=pe.to(torch.float16))["processed_prompt"]
trt_lp = lp_e(latent=latent.to(torch.float32))["image_tokens"]
trt_te = te_e(timestep=t_val.to(torch.float32))["adaln_input"]
x_trt = torch.cat([trt_lp, trt_pp], dim=1)

# Free pipeline transformer reference (we have t for PT layers)
del pipe; gc.collect(); torch.cuda.empty_cache()

print("\n" + "="*80, flush=True)
print(f"{'Step':12s} {'PT_range':26s} {'TRT_range':26s} {'Scale':>7s} {'cos_full':>9s} {'cos_img':>9s}", flush=True)
print("-"*80, flush=True)

N = 5
for li in range(N):
    with torch.no_grad():
        x_pt = t.layers[li](x_pt, attn_mask=attn, freqs_cis=freqs_cis, adaln_input=pt_te.to(torch.bfloat16))
    out = layer_trt[li](x=x_trt, attn_mask=attn.cuda(), freqs_cis=freqs_cis_trt.to(torch.float16).cuda(), adaln_input=trt_te)
    x_trt = out["output"]

    x_pt_f = x_pt.float().cuda(); x_trt_f = x_trt.float()
    pa = (abs(x_pt.max())+abs(x_pt.min()))/2; ta = (abs(x_trt.max())+abs(x_trt.min()))/2
    ratio = max(pa,ta)/(min(pa,ta)+1e-10)
    cos_full = F.cosine_similarity(x_pt_f.flatten(), x_trt_f.flatten(), dim=0)
    cos_img  = F.cosine_similarity(x_pt_f[:,:1024,:].flatten(), x_trt_f[:,:1024,:].flatten(), dim=0)
    print(f"layer_{li:02d}      [{x_pt.min():.2f},{x_pt.max():.2f}]     [{x_trt.min():.2f},{x_trt.max():.2f}]     {ratio:6.2f}x  {cos_full:9.6f}  {cos_img:9.6f}", flush=True)

# Final projection
fl = t.all_final_layer["2-1"]
with torch.no_grad():
    s = 1.0+fl.adaLN_modulation(pt_te.to(torch.bfloat16)).unsqueeze(1)
    n = fl.norm_final(x_pt.to(torch.bfloat16))*s; n=fl.linear(n); n=n[:,:1024,:]
    pt_np = n.view(n.shape[0],32,32,pH,pW,16).permute(0,5,1,3,2,4).reshape(n.shape[0],16,64,64)
trt_np = fin_e(hidden=x_trt, adaln_input=trt_te)["noise_pred"]
pt_amp=(abs(pt_np.max())+abs(pt_np.min()))/2; trt_amp=(abs(trt_np.max())+abs(trt_np.min()))/2
cos=F.cosine_similarity(pt_np.float().cuda().flatten(),trt_np.float().flatten(),dim=0)
print(f"{'final_proj':12s} [{pt_np.min():.1f},{pt_np.max():.1f}]     [{trt_np.min():.1f},{trt_np.max():.1f}]     {max(pt_amp,trt_amp)/(min(pt_amp,trt_amp)+1e-10):6.2f}x  {cos:9.6f}", flush=True)
print("\nDONE", flush=True)
