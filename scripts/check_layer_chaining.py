"""Debug: why do TRT layers produce same output when chained?"""
import torch, math, gc
import torch.distributed as dist
if not hasattr(dist, "device_mesh"):
    dist.device_mesh = type("_", (), {"DeviceMesh": type("_", (), {})})
import torch._dynamo.utils as du
if not hasattr(du, "NP_SUPPORTED_MODULES"): du.NP_SUPPORTED_MODULES = {}
import torch.nn.functional as F
o = F.scaled_dot_product_attention
F.scaled_dot_product_attention = lambda *a, **kw: o(*a, **{k:v for k,v in kw.items() if k!="enable_gqa"})
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

# Build RoPE
axes_dims=[32,48,48]; axes_lens=[1536,512,512]; seq=1152
all_f = []
for dim,length in zip(axes_dims,axes_lens):
    fq=1.0/(256.0**(torch.arange(0,dim,2).float()/dim))
    all_f.append(torch.outer(torch.arange(length).float(),fq))
pose = torch.zeros(seq,3,dtype=torch.long)
for i in range(1024): pose[i,0]=i//32; pose[i,1]=i%32
ang = torch.cat([all_f[ax][pose[:,ax]] for ax in range(3)],dim=-1)
freqs_cis_complex = torch.polar(torch.ones_like(ang), ang).unsqueeze(0)
freqs_cis_trt = torch.view_as_real(freqs_cis_complex).flatten(2).to(torch.float16).cuda()
attn = torch.ones(1, seq, dtype=torch.bool, device="cuda")

# Get t_embedder output
t_val = torch.tensor([500.0])
half_te = t.t_embedder.frequency_embedding_size // 2
freqs_t = torch.exp(-math.log(10000)*torch.arange(0,half_te,dtype=torch.float32)/half_te)
args_t = t_val[:,None].float()*freqs_t[None]
t_freq = torch.cat([torch.cos(args_t), torch.sin(args_t)], dim=-1)
with torch.no_grad():
    pt_te = t.t_embedder.mlp(t_freq.to(torch.bfloat16))

# Load TRT engines
te_e = TRTE(f"{ED}/t_embedder_fp16.engine")
pp_e = TRTE(f"{ED}/prompt_preprocessor_fp16.engine")
lp_e = TRTE(f"{ED}/latent_preprocessor_fp16.engine")
trt_te = te_e(timestep=t_val.to(torch.float32))["adaln_input"]

# Get real preprocessed x
trt_pp = pp_e(prompt_embeds=pe.to(torch.float16))["processed_prompt"]
torch.manual_seed(42); latent = torch.randn(1, 16, 64, 64)
trt_lp = lp_e(latent=latent.to(torch.float32))["image_tokens"]
x_init = torch.cat([trt_lp, trt_pp], dim=1)
print(f"x_init: [{x_init.min():.4f}, {x_init.max():.4f}]", flush=True)

# Load engines
layers = [TRTE(f"{ED}/layer_{i:02d}_fp16.engine") for i in range(5)]

# TEST 1: Feed x_init directly to each layer
print("\n=== TEST 1: Feed same x_init to each layer independently ===")
for li in range(5):
    out = layers[li](x=x_init, attn_mask=attn, freqs_cis=freqs_cis_trt, adaln_input=trt_te)
    o = out["output"]
    print(f"  layer_{li:02d}(x_init): [{o.min():.4f}, {o.max():.4f}]", flush=True)

# TEST 2: Chain layers sequentially
print("\n=== TEST 2: Chain layers (feed output of L(N) as input to L(N+1)) ===")
x_chain = x_init.clone()
for li in range(5):
    out = layers[li](x=x_chain, attn_mask=attn, freqs_cis=freqs_cis_trt, adaln_input=trt_te)
    x_new = out["output"]
    print(f"  layer_{li:02d}: in [{x_chain.min():.4f},{x_chain.max():.4f}] -> out [{x_new.min():.4f},{x_new.max():.4f}]", flush=True)
    x_chain = x_new

# TEST 3: Compare x_chain after each step via fresh inference
print("\n=== TEST 3: Verify chaining by feeding each intermediate to ALL layers ===")
chain_outs = []
x_cur = x_init.clone()
for li in range(5):
    out = layers[li](x=x_cur, attn_mask=attn, freqs_cis=freqs_cis_trt, adaln_input=trt_te)
    x_cur = out["output"]
    chain_outs.append(x_cur.clone())

# Now feed each chain_outs[i] to layer i+1 through i+4
for li in range(5):
    # Feed chain_outs[li] to layers li+1..4
    for lj in range(li+1, min(5, li+3)):
        out = layers[lj](x=chain_outs[li], attn_mask=attn, freqs_cis=freqs_cis_trt, adaln_input=trt_te)
        o = out["output"]
        expected = chain_outs[lj]
        same = (o.float() - expected.float()).abs().max() < 1e-3
        print(f"  layer_{lj:02d}(chain_out[{li}]): [{o.min():.2f},{o.max():.2f}] matches chain[{lj}]? {same}", flush=True)

print("\nDONE", flush=True)
