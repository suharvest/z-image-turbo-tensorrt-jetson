"""Check if TRT layer engines produce different outputs."""
import torch
import tensorrt as trt
ED = "/engines-v2"

class TRTE:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.ins, self.oshapes = [], {}
        self.path = path
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

# Check I/O shapes for each layer engine
for li in range(5):
    e = TRTE(f"{ED}/layer_{li:02d}_fp16.engine")
    print(f"\nlayer_{li:02d}:")
    print(f"  inputs: {e.ins}")
    print(f"  outputs: {e.oshapes}")

# Now feed same input to all 5 layers and compare outputs
print("\n\n--- Feeding same input to all 5 layers ---")
x = torch.randn(1, 1152, 3840, dtype=torch.float16, device="cuda")
attn = torch.ones(1, 1152, dtype=torch.bool, device="cuda")

# Build freqs_cis (complex -> real interleaved)
axes_dims=[32,48,48]; axes_lens=[1536,512,512]; seq=1152
all_f = []
for dim,length in zip(axes_dims,axes_lens):
    fq=1.0/(256.0**(torch.arange(0,dim,2).float()/dim))
    all_f.append(torch.outer(torch.arange(length).float(),fq))
pose = torch.zeros(seq,3,dtype=torch.long)
for i in range(1024): pose[i,0]=i//32; pose[i,1]=i%32
ang = torch.cat([all_f[ax][pose[:,ax]] for ax in range(3)],dim=-1)
freqs_cis = torch.polar(torch.ones_like(ang), ang).unsqueeze(0)
freqs_cis_trt = torch.view_as_real(freqs_cis).flatten(2).to(torch.float16).cuda()

adaln = torch.randn(1, 256, dtype=torch.float16, device="cuda")

outputs = {}
for li in range(5):
    e = TRTE(f"{ED}/layer_{li:02d}_fp16.engine")
    out = e(x=x, attn_mask=attn, freqs_cis=freqs_cis_trt, adaln_input=adaln)
    o = out[list(e.oshapes.keys())[0]]
    outputs[li] = o
    print(f"layer_{li:02d} output: [{o.min():.4f}, {o.max():.4f}] mean={o.mean():.6f} md5={hash(o.cpu().numpy().tobytes()) & 0xFFFFFFFF:08x}")

# Compare pairwise
print("\n--- Pairwise differences ---")
for i in range(5):
    for j in range(i+1, 5):
        d = (outputs[i].float() - outputs[j].float()).abs()
        print(f"  layer_{i:02d} vs layer_{j:02d}: diff mean={d.mean():.8f} max={d.max():.8f}")
