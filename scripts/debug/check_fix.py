"""Test if clone() fixes TRT layer chaining."""
import torch
import tensorrt as trt

ED = "/engines-v2"

class TRTE_fixed:
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
            t = kw[n].contiguous().cuda().clone()  # CLONE to avoid aliasing!
            self.ctx.set_tensor_address(n, t.data_ptr())
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

# Load engines with FIXED version
layers = [TRTE_fixed(f"{ED}/layer_{i:02d}_fp16.engine") for i in range(5)]

# Test random input chain
torch.manual_seed(42)
x = torch.randn(1, 1152, 3840, dtype=torch.float16, device="cuda")
print(f"Input: [{x.min():.4f}, {x.max():.4f}]", flush=True)

x_chain = x.clone()
for li in range(5):
    out = layers[li](x=x_chain, attn_mask=attn, freqs_cis=freqs_cis_trt, adaln_input=adaln)
    x_new = out["output"]
    diff = (x_new.float() - x_chain.float()).abs()
    print(f"layer_{li:02d}: in=[{x_chain.min():.2f},{x_chain.max():.2f}] -> out=[{x_new.min():.2f},{x_new.max():.2f}] diff={diff.mean():.6f} is_identity={diff.mean()<1e-5}", flush=True)
    x_chain = x_new

# Also verify: direct call with the chain outputs
print("\n--- Verification: direct call with each intermediate ---", flush=True)
for li, chain_out in enumerate([x.clone(), layers[0](x=x, attn_mask=attn, freqs_cis=freqs_cis_trt, adaln_input=adaln)["output"]]):
    print(f"  chain_out[{li}] range: [{chain_out.min():.2f},{chain_out.max():.2f}]", flush=True)

print("\nDONE", flush=True)
