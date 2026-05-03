#!/bin/bash
set -e

docker run --rm --privileged --network=host \
  -v /usr/lib/aarch64-linux-gnu:/host-libs:ro \
  -v /etc/alternatives:/etc/alternatives:ro \
  -v /usr/local/cuda-12.6:/usr/local/cuda:ro \
  -v /home/harvest/.local/lib/python3.10/site-packages/nvidia:/usr/local/nvidia-pip:ro \
  -e LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:/usr/local/cuda/nvvm/lib64:/host-libs:/host-libs/tegra:/host-libs/openblas-pthread:/usr/local/nvidia-pip/cusparselt/lib" \
  -v /home/harvest/models:/models:ro \
  -v /home/harvest/z-image-output:/output \
  --shm-size=4g \
  z-image-jetson:latest bash -c '
pip3 install "numpy<2" 2>&1 | tail -2

python3 << PYEOF
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("arch:", torch.cuda.get_arch_list())
a = torch.tensor([1.0, 2.0, 3.0]).cuda()
b = a * 2
print("CUDA OK:", b)
PYEOF

python3 << PYEOF
import torch, os
from diffusers import ZImagePipeline

print("Loading model...")
pipe = ZImagePipeline.from_pretrained(
    "/models/z-image-turbo-fp8-diffusers",
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    max_memory={0: "10GB"},
)
print("Generating...")
image = pipe(
    prompt="A cute orange tabby cat sitting on a sunny windowsill, digital art",
    height=512,
    width=512,
    num_inference_steps=4,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
os.makedirs("/output", exist_ok=True)
image.save("/output/output.png")
print("IMAGE SAVED!")
PYEOF
'
echo "EXIT=$?"
