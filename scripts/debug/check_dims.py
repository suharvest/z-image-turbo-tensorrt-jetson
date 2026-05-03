"""Check transformer head dimensions."""
import torch, torch.distributed as dist
if not hasattr(dist, 'device_mesh'):
    dist.device_mesh = type('_', (), {'DeviceMesh': type('_', (), {})})
import torch._dynamo.utils as du
if not hasattr(du, 'NP_SUPPORTED_MODULES'):
    du.NP_SUPPORTED_MODULES = {}
import torch.nn.functional as F
o = F.scaled_dot_product_attention
F.scaled_dot_product_attention = lambda *a, **kw: o(*a, **{k:v for k,v in kw.items() if k!='enable_gqa'})
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from diffusers import ZImagePipeline
pipe = ZImagePipeline.from_pretrained(
    '/models/z-image-turbo-fp8-diffusers',
    torch_dtype=torch.bfloat16,
    device_map='balanced',
    max_memory={0: '12GB'},
)
t = pipe.transformer
print('n_heads:', t.config.n_heads)
print('n_kv_heads:', t.config.n_kv_heads)
print('dim:', t.config.dim)
print('head_dim:', t.config.dim // t.config.n_heads)
print('total rope dim:', sum(t.config.axes_dims))

# Check the actual query shape from layer 0 attention
layer0 = t.layers[0]
print('q_proj weight:', layer0.self_attn.q_proj.weight.shape)
print('k_proj weight:', layer0.self_attn.k_proj.weight.shape)
print('v_proj weight:', layer0.self_attn.v_proj.weight.shape)
