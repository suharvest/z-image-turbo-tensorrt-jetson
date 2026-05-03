#!/usr/bin/env python3
# Z-Image-Turbo 下载脚本 (使用国内镜像)

import os
import sys

# 设置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

def main():
    model_id = "Tongyi-MAI/Z-Image-Turbo"
    local_dir = os.path.expanduser("~/models/z-image-turbo")

    print("=" * 50)
    print("Z-Image-Turbo 模型下载脚本")
    print("(使用 hf-mirror.com 镜像)")
    print("=" * 50)
    print(f"模型: {model_id}")
    print(f"目录: {local_dir}")
    print(f"大小: ~12GB")
    print(f"镜像: {os.environ.get('HF_ENDPOINT')}")
    print()

    # 创建目录
    os.makedirs(local_dir, exist_ok=True)
    print("目录已创建")

    print("开始下载...")
    print("时间:", os.popen('date').read().strip())
    print()

    try:
        path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            max_workers=4,
        )
        print()
        print("=" * 50)
        print("下载完成!")
        print("=" * 50)
        print(f"模型位置: {path}")

        # 显示文件大小
        total_size = os.popen(f'du -sh {local_dir}').read().strip()
        print(f"总大小: {total_size}")

        # 显示关键文件
        files = os.listdir(local_dir)
        print(f"文件数: {len(files)}")
        print("关键文件检查:")
        required = ['model_index.json', 'scheduler', 'transformer', 'vae']
        for r in required:
            exists = r in files or os.path.exists(os.path.join(local_dir, r))
            print(f"  {r}: {'✅' if exists else '❌'}")

    except Exception as e:
        print()
        print("=" * 50)
        print("下载失败!")
        print("=" * 50)
        print(f"错误: {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()