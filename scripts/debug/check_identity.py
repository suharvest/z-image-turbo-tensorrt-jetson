"""Test if TRT layers act as identity for certain inputs but not others."""
import torch, math
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
import tensorrt as trt, gc

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
adaln = torch.randn(1, 256, dtype=torch.float16, device="cuda")

layers = [TRTE(f"{ED}/layer_{i:02d}_fp16.engine") for i in range(5)]

# Test inputs of different scales
scales = [0.01, 0.1, 1.0, 10.0, 100.0]
print(f"{'Input scale':15s} {'layer_00: in->out range':45s} {'layer_01: in->out range':45s}")
print("-" * 105)

for s in scales:
    torch.manual_seed(42)
    x = torch.randn(1, 1152, 3840, dtype=torch.float16, device="cuda") * s
    in_range = f"[{x.min():.2f},{x.max():.2f}]"

    out0 = layers[0](x=x, attn_mask=attn, freqs_cis=freqs_cis_trt, adaln_input=adaln)["output"]
    diff0 = (out0.float() - x.float()).abs()
    out0_range = f"[{out0.min():.2f},{out0.max():.2f}] diff={diff0.mean():.4f}"

    out1 = layers[1](x=x, attn_mask=attn, freqs_cis=freqs_cis_trt, adaln_input=adaln)["output"]
    diff1 = (out1.float() - x.float()).abs()
    out1_range = f"[{out1.min():.2f},{out1.max():.2f}] diff={diff1.mean():.4f}"

    print(f"{'scale='+str(s):15s} {in_range} -> {out0_range:45s} {out1_range}", flush=True)

# Also test: feed the SAME input through chained vs independent layers
print("\n--- Chain test with random input ---")
torch.manual_seed(42)
x = torch.randn(1, 1152, 3840, dtype=torch.float16, device="cuda")
print(f"Input: [{x.min():.4f}, {x.max():.4f}]")
x_chain = x.clone()
for li in range(5):
    out = layers[li](x=x_chain, attn_mask=attn, freqs_cis=freqs_cis_trt, adaln_input=adaln)
    x_chain = out["output"]
    out_direct = layers[li](x=x.clone(), attn_mask=attn, freqs_cis=freqs_cis_trt, adaln_input=adaln)["output"]
    diff_chain_vs_direct = (x_chain.float() - out_direct.float()).abs()
    print(f"  layer_{li:02d}: chain=[{x_chain.min():.2f},{x_chain.max():.2f}] direct=[{out_direct.min():.2f},{out_direct.max():.2f}] diff={diff_chain_vs_direct.mean():.6f}", flush=True)

print("\nDONE", flush=True)
