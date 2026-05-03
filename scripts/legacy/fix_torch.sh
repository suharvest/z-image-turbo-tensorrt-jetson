#!/bin/bash
set -e
echo "Downloading Jetson PyTorch..."
cd /tmp
wget -q "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl" -O torch_jetson.whl
echo "Downloaded: $(ls -lh torch_jetson.whl | awk '{print $5}')"

echo "Installing in container and testing..."
docker run --rm --privileged --network=host \
  -v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra:ro \
  -v /usr/local/cuda-12.6:/usr/local/cuda:ro \
  -e LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:/usr/local/cuda/nvvm/lib64:/usr/lib/aarch64-linux-gnu/tegra" \
  -v /tmp/torch_jetson.whl:/tmp/torch_jetson.whl:ro \
  -v /tmp/test_kernel.py:/workspace/test_kernel.py:ro \
  z-image-turbo:latest bash -c "
pip3 install --no-deps /tmp/torch_jetson.whl
echo '--- CUDA TEST ---'
python3 /workspace/test_kernel.py
"
echo "DONE"
