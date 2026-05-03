#!/bin/bash
# Z-Image-Turbo 模型准备脚本 (只下载，不推理)
# 用法: bash prepare_z_image.sh

set -e

MODEL_NAME="Tongyi-MAI/Z-Image-Turbo"
MODEL_DIR="$HOME/models/z-image-turbo"
EXPECTED_SIZE="12GB"

echo "=========================================="
echo "Z-Image-Turbo 模型准备脚本"
echo "=========================================="
echo "目标设备: Jetson Orin NX"
echo "模型大小: ~${EXPECTED_SIZE}"
echo "存放目录: ${MODEL_DIR}"
echo ""

# 1. 检查磁盘空间
echo "[1/5] 检查磁盘空间..."
AVAIL_GB=$(df -BG $HOME | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAIL_GB" -lt 15 ]; then
    echo "❌ 磁盘空间不足，需要至少 15GB，当前只有 ${AVAIL_GB}GB"
    exit 1
fi
echo "✅ 磁盘空间充足: ${AVAIL_GB}GB 可用"

# 2. 安装 pip
echo ""
echo "[2/5] 安装 pip..."
if ! python3 -m pip --version &>/dev/null; then
    echo "安装 python3-pip..."
    sudo apt-get update -qq
    sudo apt-get install -y python3-pip python3-venv
    echo "✅ pip 已安装"
else
    echo "✅ pip 已存在"
fi

# 3. 安装 huggingface-cli
echo ""
echo "[3/5] 安装 huggingface-cli..."
pip3 install --upgrade huggingface_hub
echo "✅ huggingface-cli 已安装"

# 4. 创建模型目录
echo ""
echo "[4/5] 创建模型目录..."
mkdir -p "$MODEL_DIR"
echo "✅ 目录已创建: $MODEL_DIR"

# 5. 下载模型
echo ""
echo "[5/5] 下载 Z-Image-Turbo 模型..."
echo "模型名称: $MODEL_NAME"
echo "下载开始时间: $(date)"
echo ""

# 使用 huggingface-cli 下载
huggingface-cli download "$MODEL_NAME" \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False

echo ""
echo "下载完成时间: $(date)"
echo ""

# 验证下载
echo "=========================================="
echo "验证下载文件..."
echo ""
DOWNLOADED_SIZE=$(du -sh "$MODEL_DIR" | cut -f1)
echo "下载大小: $DOWNLOADED_SIZE"

# 检查关键文件
REQUIRED_FILES=(
    "model_index.json"
    "scheduler/scheduler_config.json"
    "transformer/config.json"
    "vae/config.json"
)

MISSING=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$MODEL_DIR/$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (缺失)"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "⚠️ 部分文件缺失，请重新下载"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Z-Image-Turbo 模型准备完成!"
echo "=========================================="
echo ""
echo "模型位置: $MODEL_DIR"
echo "模型大小: $DOWNLOADED_SIZE"
echo ""
echo "后续步骤:"
echo "  1. 当内存空闲时，可使用以下命令测试:"
echo "     python3 -c 'from diffusers import ZImagePipeline; print(\"ok\")'"
echo ""
echo "  2. 加载模型:"
echo "     pipe = ZImagePipeline.from_pretrained(\"$MODEL_DIR\")"
echo ""
echo "=========================================="