# 把 Z-Image-Turbo 跑上 Jetson：一次 TensorRT 适配记录

> 这篇文章记录我们把 [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) 适配到 NVIDIA Jetson Orin NX 的过程。目标不是做一个实验室 demo，而是把它变成一个别人可以下载、部署、调用的边缘端图像生成服务。

## 背景

Z-Image-Turbo 是一个 6B 规模的 DiT 图像生成模型。它的生成效果不错，但原始 PyTorch / diffusers 运行方式对边缘设备不太友好：

- 运行镜像很大，包含 PyTorch、diffusers、transformers 等完整生态。
- 显存和内存压力高，Jetson 上很容易 OOM。
- 生成链路里有很多 Python 层调度和模型加载开销。
- 模型权重、ONNX、TensorRT engine 都很大，不能简单塞进 GitHub 仓库。

所以我们的目标很直接：在 Jetson 上用 TensorRT 跑通 Z-Image-Turbo，并且把部署方式整理成一个可复现、可调用、可开源的工程。

## 先用一张图理解 Z-Image 大致怎么工作

Z-Image-Turbo 可以简单理解成一个“在 latent 空间里画图”的 diffusion transformer。

如果是 text-to-image，流程大致是：

1. prompt 先经过 tokenizer 和 text encoder，变成模型能理解的文本特征。
2. 随机噪声 latent 作为起点。
3. DiT denoise 主体按 step 一轮轮把噪声变成更清晰的图像 latent。
4. VAE decoder 把 latent 解码成 RGB 图片。

如果是 image-to-image，会多一条参考图路径：

1. 参考图先经过 VAE encoder，变成 latent。
2. 按 `strength` 对这个 latent 加噪。
3. 再进入同一个 denoise 主体，结合 prompt 改写图像。

这次 TensorRT 适配做的事情，就是把这些关键模块从 PyTorch runtime 里拆出来，变成一组可以在 Jetson 上加载的 TensorRT engine。

![Z-Image-Turbo TensorRT runtime pipeline](../media/article/z-image-tensorrt-pipeline.svg)

## 最后做成了什么

最终得到的是一个 no-PyTorch 的 TensorRT runtime：

- 支持 384x384 和 512x512。
- 支持 text-to-image。
- 支持 image-to-image，也就是参考图加文字 prompt。
- 支持 HTTP API 调用。
- 支持把生成结果保存到用户指定的 host 目录。
- 支持通过 `/outputs/<file>.png` 直接下载生成图片。
- Runtime Docker 镜像约 428MB。
- 大的 TensorRT engines 独立发布到 Hugging Face artifact repo。

部署后调用方式类似这样：

```bash
curl -X POST http://<jetson-ip>:8000/generate_json \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cute orange tabby cat sitting on a sunny windowsill, photorealistic",
    "num_steps": 4,
    "seed": 42,
    "output_name": "cat.png"
  }'
```

返回结果里会有图片地址：

```json
{
  "success": true,
  "image_path": "/output/cat.png",
  "image_url": "/outputs/cat.png",
  "elapsed_seconds": 118.648,
  "trt_seconds": 78.3
}
```

然后直接下载：

```bash
curl http://<jetson-ip>:8000/outputs/cat.png -o cat.png
```

输出目录不是写死的。启动服务时可以用 `OUTPUT_DIR_HOST` 指定 host 侧目录，例如：

```bash
OUTPUT_DIR_HOST=/data/z-image-output \
UPLOAD_DIR_HOST=/data/z-image-input \
scripts/run/run_3drope_no_torch_api.sh
```

容器里统一挂载为 `/output` 和 `/uploads`，但 host 上放在哪里由用户自己决定。

## 真实输出长什么样

这类适配最容易让人误判的地方是：程序跑完了、文件也保存了，但图像不一定是对的。下面这些图都是适配过程中真实生成过的结果。

### 失败样例 1：engine 能跑完，但 denoise 语义错了

![Wrong noise refiner branch output](../media/article/failure-noise-refiner-wrong-branch.png)

这张图的问题不是“审美不好”，而是模型基本没有完成有效 denoise，只剩下彩色噪声。原因是 `noise_refiner` 的 ONNX 导出分支和真实 PyTorch 调用路径不一致。

Z-Image basic mode 实际走的是全局 `adaln_input` 调制；旧 TensorRT export 走成了 `noise_mask / t_noisy / t_clean` 分支。engine 可以构建，也可以推理，但语义已经错了。

### 失败样例 2：VAE decoder 数值异常，输出黑图

![VAE FP16 black output](../media/article/failure-vae-fp16-black.png)

这张黑图来自早期 VAE TensorRT 版本。单看服务状态，它也可能是“成功生成了 png”；但实际是 VAE decoder 的 FP16 构建路径在 Orin NX 上出现数值异常，最终 RGB 几乎全被压成黑色。

后续处理方式是改用 BF16 VAE decoder，并在调试阶段加入 tensor 统计检查，避免只靠肉眼看最终图。

### 修复后：512 text-to-image 正常输出

![Fixed TensorRT 512 output](../media/article/success-512-refiner-fixed.png)

修复 `noise_refiner` 导出输入后，512 TensorRT 输出恢复成正常图像：主体、窗台、光照和毛发结构都能稳定生成。

### 当前 no-PyTorch runtime：512 输出样例

![Final no-PyTorch 512 output](../media/article/success-512-final-runtime.png)

这是后续迁移到 no-PyTorch runtime 后的 512 输出样例。它不是为了和云端大 GPU 比速度，而是验证同一套 TensorRT engines、VAE、text encoder、scheduler 和 API wrapper 可以在 Jetson 上闭环跑通。

## 我们具体做了哪些事情

### 1. 拆分模型，把大模型变成一组 TensorRT engine

Z-Image-Turbo 的主体是 30 层 transformer。直接把整个模型一次性导成一个巨大的 engine，不适合 Jetson：

- engine 太大。
- 构建和加载都很重。
- 中间调试困难。
- 某一层出错时很难定位。

所以我们采用了分层导出：

- transformer 30 层逐层导出。
- prompt preprocessor / latent preprocessor / final projection 单独导出。
- context refiner / noise refiner 单独导出。
- VAE decoder / encoder 单独导出。
- text encoder 后面改成分组 engine。

这样做的好处是：每一层都可以单独比对、单独替换、单独缓存。坏处是 runtime 要负责调度更多 engine，但这个复杂度是可控的。

### 2. 做逐层对比，先保证“图像是对的”

刚开始不是性能问题，而是正确性问题。

生成图像曾经出现过重影、模糊、多曝光一样的结果。这个时候如果只看最终图片，很难知道是哪一层错了。

所以我们做了逐层对比：

- PyTorch 输出作为 reference。
- ONNX 输出和 PyTorch 对比。
- TensorRT 输出和 ONNX / PyTorch 对比。
- 每一层检查误差和 tensor 统计。

最后定位到一个关键问题：`noise_refiner` 的导出分支和实际推理调用不一致。

PyTorch basic mode 里实际使用的是全局 `adaln_input` 分支，而旧的 TensorRT export 走的是另一个 `noise_mask / t_noisy / t_clean` 分支。结果就是 engine 能跑，但语义不对，最终图像自然也不对。

修复方式是重新导出 `noise_refiner_00/01`，让输入和 PyTorch 实际调用保持一致：

- `x`
- `attn_mask`
- `freqs_cis`
- `adaln_input`

这个修完后，多曝光和模糊问题消失，TensorRT 生成结果才真正变成“对的图”。

### 3. 加 384 和 512 两种固定分辨率

TensorRT engine 通常是静态 shape。为了稳定和性能，我们没有做一个完全动态分辨率 engine，而是分别准备：

- 384x384 engine
- 512x512 engine

这意味着部署时选择一个分辨率模式：

```bash
RESOLUTION=384
```

或：

```bash
RESOLUTION=512
```

固定分辨率的好处是简单、稳定、可控。代价是如果要支持更多尺寸，就需要额外导出和构建对应 engine。

### 4. 把 VAE 和 text encoder 也迁移到 TensorRT

一开始我们主要加速 denoise transformer。后来继续把 VAE 和 text encoder 也迁移到 TensorRT。

这一步很重要，因为如果 runtime 里还保留 PyTorch，只是 transformer 用 TensorRT，那么：

- 镜像还是很大。
- PyTorch 会常驻内存。
- Jetson 上可用于缓存 engine 的空间会变少。
- 冷启动和释放资源都更麻烦。

最终 no-PyTorch runtime 只保留：

- TensorRT Python
- CUDA Runtime，通过 `ctypes` 调用
- NumPy
- Pillow
- tokenizers

不再导入：

- PyTorch
- diffusers
- transformers

这也是为什么镜像可以压到约 428MB。

### 5. 做 engine cache 和 buffer 复用

TensorRT engine 加载本身有成本。每一层用完就卸载，内存省，但速度慢；全部缓存，速度快，但容易 OOM。

所以我们做了两件事：

第一，允许配置缓存多少层：

```bash
MAX_CACHED_LAYERS=18
```

第二，在 no-PyTorch runtime 里复用 layer 输出 buffer，避免每层都重复申请大块 CUDA 内存。

这让 384 模式在 Orin NX 16GB 上可以缓存全部 30 个 denoise layer engine。512 模式内存压力更高，目前验证的是缓存 18 层。

### 6. 做 HTTP API，而不是只给一个脚本

为了让别人更容易用，我们最后加了一层 FastAPI wrapper。

接口包括：

- `GET /health`
- `POST /generate`
- `POST /generate_json`
- `GET /outputs/<file>.png`

`/generate` 支持 multipart 上传参考图；`/generate_json` 适合调用方已经把图片放到容器可见路径的场景。

API 内部一次只跑一个请求。原因很简单：图像生成在 Jetson 上是重任务，多个请求同时跑很容易 OOM。现在的设计是：

- HTTP 服务接收请求。
- 请求排队。
- 每次只启动一个 TensorRT 子进程生成图片。
- 子进程退出后释放 CUDA / TensorRT 资源。

这个设计牺牲了一点并发，但换来了稳定性。

### 7. 把大文件从代码仓库里拆出去

TensorRT engines 太大了：

- 384 engines 约 12.8GB
- 512 engines 约 13.0GB
- split text encoder engines 约 16.1GB

这些文件不能直接放进普通 Git 仓库。我们最后把源码和 artifacts 分开：

- GitHub / Git 仓库放源码、导出脚本、运行脚本、manifest。
- Hugging Face repo 放生成好的 TensorRT engines。

artifact repo：

```text
harvestsu/z-image-turbo-jetson-trt-artifacts
```

用户可以下载：

```bash
hf download harvestsu/z-image-turbo-jetson-trt-artifacts \
  --local-dir "$HOME/models/z-image-trt-artifacts"
```

然后 launcher 默认就会去这个目录结构里找 engine。

## 遇到的坑

### 坑 1：能跑不等于图是对的

TensorRT engine 能成功构建、能跑完、能输出图片，并不代表结果正确。

这次最典型的问题就是 `noise_refiner`。它不是崩溃，也不是输出 NaN，而是输出了一张“看起来像图但质量明显不对”的图片。

这种问题只能靠逐层对比解决。最终经验是：适配生成模型时，不要只做端到端 smoke test，必须保留中间层对比工具。

### 坑 2：分支导出必须和真实调用路径一致

很多 PyTorch 模型里有条件分支。有的分支只在训练用，有的分支只在某种 scheduler 或 pipeline 下用。

导出 ONNX 时如果走错分支，engine 仍然可能构建成功，但推理语义已经偏了。

这次 `noise_refiner` 就是这个问题。修复后我们把导出输入对齐到真实 runtime 路径，而不是只追求“这个模块能导出”。

### 坑 3：Jetson 上内存比算力更早成为瓶颈

很多优化一开始看起来是性能问题，最后发现本质是内存问题。

例如：

- 缓存更多 layer engine 可以减少加载开销，但会增加常驻内存。
- img2img 比 text2img 多一次 VAE encode，内存压力更大。
- PyTorch / diffusers / transformers 常驻会挤占本来可以给 TensorRT engine cache 的空间。

去掉 PyTorch 之后，内存压力明显下降，384 模式才能缓存全部 30 层。

### 坑 4：img2img 的 strength 不是“效果强度”这么简单

img2img 里 `strength` 控制的是参考图被加噪到多深，以及后面实际 denoise 的步数。

简单说：

- strength 低：更保留原图，变化小。
- strength 高：更听 prompt，变化大。
- 在固定 `num_steps` 下，strength 也会影响实际 denoise step 数。

例如 8 steps、strength 0.65 时，实际 denoise steps 是 5。这个会直接影响速度和最终效果。

### 坑 5：大文件上传不能用普通思路

一开始用普通 `hf upload` 上传大 engine 目录，尾部卡住过。后来改用：

```bash
hf upload-large-folder
```

这个命令会保存断点状态，适合多 GB 目录。对这种 artifact 发布流程来说，它比单次 commit 式上传稳定很多。

### 坑 6：不要把 engine 当成通用模型格式

TensorRT engine 不是 ONNX，也不是 PyTorch checkpoint。它和硬件、TensorRT、CUDA、JetPack 都强相关。

所以 artifact 目录必须带 target 信息，例如：

```text
engines/orin-nx-jp6-trt10.3/
```

否则用户很容易在不匹配的设备或 TensorRT 版本上加载失败。

## 最终效果

在 Jetson Orin NX 16GB 上，当前验证结果如下。

### Runtime 镜像

| 项目 | 结果 |
|---|---:|
| Docker image | `sensecraft-missionpack.seeed.cn/solution/z-image-jetson-no-torch:latest` |
| Image size | 428MB |
| Runtime dependency | TensorRT + CUDA Runtime + NumPy + Pillow + tokenizers |
| PyTorch | 不需要 |
| diffusers / transformers | 不需要 |

### 生成能力

| 能力 | 状态 |
|---|---|
| 384 text-to-image | 已验证 |
| 512 text-to-image | 已验证 |
| 384 image-to-image | 已验证 |
| 512 image-to-image | 已验证 |
| HTTP API | 已验证 |
| 输出文件服务 | 已验证 |

### 性能数据

| 模式 | 设置 | 总耗时 | TensorRT denoise |
|---|---|---:|---:|
| 384 text-to-image | 4 steps, cache 30 layers | 92.8s | 56.2s |
| 384 image-to-image | 8 steps, strength 0.65, cache 18 layers | 123.1s | 86.9s |
| 512 text-to-image | 4 steps, cache 18 layers | 117.4s | 80.1s |
| 512 image-to-image | 8 steps, strength 0.65, cache 18 layers | 129.7s | 91.6s |
| 512 API text-to-image | 4 steps | 118.648s | 78.3s |
| 512 API image-to-image | 4 steps, strength 0.65 | 86.276s | 46.4s |

这些数字不是云端 GPU 的速度，但它们说明了一件事：6B 级别的图像生成模型可以被整理成一个能在 Jetson 上稳定运行的 TensorRT 服务。

## 这件事的价值

这次适配最有价值的地方，不只是“跑通了一个模型”，而是形成了一套边缘端生成模型部署方法：

1. 大模型拆成可验证的小 engine。
2. 用逐层对比保证正确性。
3. 把 VAE、text encoder、denoise 主体逐步迁到 TensorRT。
4. 去掉 PyTorch runtime，降低镜像和内存占用。
5. 通过 engine cache 和 buffer 复用平衡速度与内存。
6. 用 HTTP API 把 demo 变成服务。
7. 用独立 artifact repo 发布大文件，让开源仓库保持轻量。

对后来要把其他生成模型搬到 Jetson、Orin Nano、Orin NX 或类似边缘设备的人来说，这套流程可以复用。

## 还有什么可以继续优化

目前还有继续提升的空间：

- 用 C++ runtime 减少 Python 调度开销。
- 做更大的 layer group engine，减少 engine 调用次数。
- 针对 BF16 / FP16 cast 写更轻量的 CUDA kernel。
- 针对 8GB 设备单独做更激进的内存策略。
- 增加更多分辨率档位，但需要对应构建新的 TensorRT engines。

不过现在这个版本已经从“能跑的实验”变成了“可以部署和复现的工程”。这也是这次 TensorRT 适配最重要的结果。
