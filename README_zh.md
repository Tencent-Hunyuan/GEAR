## GEAR: Guided End-to-End AutoRegression for Image Synthesis<br><sub>官方 PyTorch 实现</sub>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-README-2f6df0.svg" alt="English"></a>
  <a href="README_zh.md"><img src="https://img.shields.io/badge/中文-说明-c0392b.svg" alt="中文"></a>
  <a href="https://linb203.github.io/gear/"><img src="https://img.shields.io/badge/🏠-Homepage-1a73e8.svg" alt="Homepage"></a>
  <a href=""><img src="https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg?logo=arxiv" alt="arXiv"></a>
  <a href=""><img src="https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg" alt="HF paper"></a>
  <a href="https://huggingface.co/collections/BinLin203"><img src="https://img.shields.io/badge/🤗-Models-blue.svg" alt="HF models"></a>
</p>

<p align="center">
  <a href="https://linb203.github.io/">Bin Lin</a><sup>1,2,*</sup>,
  Zheyuan Liu<sup>1,2,*</sup>,
  <a href="https://chenguolin.github.io/">Chenguo Lin</a><sup>1</sup>,
  <a href="https://ephemeral182.github.io/">Sixiang Chen</a><sup>2,*</sup>,
  <a href="https://scholar.google.com/citations?user=bbgjlg0AAAAJ">Yunyang Ge</a><sup>1,2,*</sup>,
  <br>
  <a href="https://lyl1015.github.io/">Yunlong Lin</a>,
  <a href="https://scholar.google.com/citations?user=nF_klRIAAAAJ">Jianwei Zhang</a><sup>2</sup>,
  Miles Yang<sup>2</sup>,
  <a href="https://scholar.google.com/citations?user=igtXP_kAAAAJ">Zhao Zhong</a><sup>2</sup>,
  <a href="https://research.cs.washington.edu/istc/lfb/">Liefeng Bo</a><sup>2</sup>,
  <a href="https://yuanli2333.github.io/">Li Yuan</a><sup>1,&dagger;</sup>
</p>

<p align="center">
  <sup>1</sup> 北京大学 &nbsp;&nbsp; <sup>2</sup> 腾讯混元
  <br>
  <sup>*</sup> 于腾讯混元实习期间完成 &nbsp;&nbsp; <sup>&dagger;</sup> 通讯作者
</p>

<p align="center">
  <img src="https://s41.ax1x.com/2026/06/25/pmtXRN4.png" width="100%" alt="GEAR teaser"/>
</p>

> 🚀 **GEAR 将 VQ 分词器与自回归（AR）生成器端到端联合训练** —— 关键是让分词器感知到来自下游生成器的梯度，从而使它产生的 token 更易被预测。ImageNet gFID 收敛最高快约 **10×**，并可即插即用地泛化到 **VQ / LFQ / IBQ** 与文生图。

### ✨ 亮点

- 🔗 **引导式端到端。** soft 软分配桥让 AR 去 *引导* 分词器，恰好在直通估计器坍塌之处成功；next-token 损失永远不碰分词器。
- 🔄 **对齐转移到了 AR。** 与扩散侧 REPA（REPA-E / VA-VAE）相反：分词器变得 *更不* 像 DINOv2、熵更低，而 AR 的逐 patch 特征与 DINOv2 贴合得更近，同时重建保持不变。
- ⚡ **更快、更好。** ImageNet gFID 收敛快约 10×；在 GPIC 文生图上，全新 AR 在冻结分词器上达到基线 NTP 损失快 **2.5×**、REPA 损失快 **11.1×**，且 B / L / XL 各规模 gFID 更优。
- 🧩 **通用、即插即用。** 适用于 VQVAE / LFQ / IBQ，覆盖类别条件 ImageNet 与文生图 —— 冻结调好的分词器即可放进标准流程。

<details>
<summary><b>🔬 工作原理</b></summary>

<br>

端到端训练 AR 的难点在于 VQ 索引是不可导的 `argmax`（用直通估计器会坍塌，gFID≈105）。GEAR 用对码本的 **双重读出** 解决：一条 **hard** 独热分支训练 AR，一条可导的 **soft** 分支 `softmax(-d/τ)` 携带 REPA 损失回传，**只更新分词器**。两者的优化保持解耦，共用一个系数 λ、互不串扰：

$$\theta_{\mathrm{tok}} \leftarrow \theta_{\mathrm{tok}} - \eta \nabla_{\theta_{\mathrm{tok}}} \left( \mathcal{L}_{\mathrm{VQ}} + \lambda \mathcal{L}^{\mathrm{s}}_{\mathrm{align}} \right)$$

$$\theta_{\mathrm{AR}} \leftarrow \theta_{\mathrm{AR}} - \eta \nabla_{\theta_{\mathrm{AR}}} \left( \mathcal{L}_{\mathrm{NTP}} + \lambda \mathcal{L}^{\mathrm{h}}_{\mathrm{align}} \right)$$

分词器损失为标准 VQ 目标（重建 + LPIPS + GAN + 熵 + commit）加上 **soft** 对齐项；AR 由 NTP 加 **hard** REPA 项训练。

</details>

---

## 目录
- [安装](#安装)
- [预训练权重](#预训练权重)
- [数据准备](#数据准备)
- [训练](#训练)
- [推理](#推理)
- [评测](#评测)
- [引用](#引用)

---

## 安装

```bash
conda create -n gear python=3.13 -y
conda activate gear

# 1) 先装 PyTorch，按你的 CUDA 版本匹配（示例：CUDA 12.8）
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 2) 其余依赖
pip install -r requirements.txt
```

上面的 `gear` 环境覆盖训练、推理以及仓库内置的 GPIC FD-DINOv2 评测。外部 benchmark 打分器
（GenEval、DPG-Bench、WISE judge，以及 ADM ImageNet 评估器）各自需要 **独立的** conda 环境，
请在运行这些 benchmark 前，按 [评测](#评测) 中链接的各 benchmark README 创建对应环境。

---

## 预训练权重

GEAR 的核心贡献是 **tokenizer**，所以我们开源的也正是它。请用 [训练脚本](#训练) 在冻结的
GEAR tokenizer 上训练你自己的 AR。

对每种公开量化器（VQ-16 / LFQ-16 / IBQ-16，均为 16384 码本），我们在
[🤗 Hub](https://huggingface.co/BinLin203) 上发布两个 tokenizer：

- **Warm-up** —— 一次简短的重建微调，仅用于恢复公开权重缺失的 GAN 判别器。它是
  LlamaGen-REPA 与 GEAR 共同的起点（基线）。
- **GEAR** —— 同一 tokenizer 经过与 AR **端到端** 微调后的版本。

| 量化器 | Warm-up（基线） | GEAR（端到端） |
|---|---|---|
| VQ-16 | [`Warmup-VQ`](https://huggingface.co/BinLin203/Warmup-VQ) · `vq-with-gan.pt` | [`GEAR-VQ`](https://huggingface.co/BinLin203/GEAR-VQ) · `gear-vq.pt` |
| LFQ-16 | [`Warmup-LFQ`](https://huggingface.co/BinLin203/Warmup-LFQ) · `lfq-with-gan.pt` | [`GEAR-LFQ`](https://huggingface.co/BinLin203/GEAR-LFQ) · `gear-lfq.pt` |
| IBQ-16 | [`Warmup-IBQ`](https://huggingface.co/BinLin203/Warmup-IBQ) · `ibq-with-gan.pt` | [`GEAR-IBQ`](https://huggingface.co/BinLin203/GEAR-IBQ) · `gear-ibq.pt` |

```bash
huggingface-cli download BinLin203/GEAR-VQ --local-dir ckpts/GEAR-VQ
```

### 重建质量（ImageNet 验证集）

预热（warm-up）与端到端微调（GEAR）后的 tokenizer，重建性能都与原始预训练权重相当。
*Original* 为公开发布版本。

| 量化器 | 设置 | rFID↓ | PSNR↑ | SSIM↑ |
|---|---|---|---|---|
| VQ-16 | Original | 2.19 | 20.79 | 0.55 |
| | Warm-up | 1.72 | 21.06 | 0.57 |
| | **GEAR** | **1.64** | 20.78 | 0.56 |
| LFQ-16 | Original | 2.82 | 21.47 | 0.58 |
| | Warm-up | 2.42 | 20.97 | 0.56 |
| | **GEAR** | **2.13** | 20.48 | 0.55 |
| IBQ-16 | Original | 2.23 | 21.23 | 0.58 |
| | Warm-up | 1.97 | 21.18 | 0.58 |
| | **GEAR** | **1.72** | 20.92 | 0.57 |

> 所有行均使用 **bicubic** 缩放以保证可比。插值方式会影响结果（LFQ / IBQ 官方数值用的是
> bilinear，二者有差距），完整的 bilinear / bicubic 对照见论文附录。

### 原始 tokenizer（用于复现 warm-up）

上表的 *Original* 即这些公开预训练权重；想自己复现 warm-up，请从它们开始：

- VQ：[`vq_ds16_c2i.pt`](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt)（LlamaGen）
- IBQ：[`IBQ_pretrain_16384.ckpt`](https://huggingface.co/TencentARC/IBQ-Tokenizer-16384-Pretrain/blob/main/IBQ_pretrain_16384.ckpt)（TencentARC）
- LFQ：[`pretrain256_16384.ckpt`](https://huggingface.co/TencentARC/Open-MAGVIT2-Tokenizer-16384-Pretrain/blob/main/pretrain256_16384.ckpt)（Open-MAGVIT2）

---

## 数据准备

- **ImageNet-1K**（分词器 + AR 训练）：标准 `ImageFolder` 布局
  `<root>/<synset>/<file>.JPEG`。数据增强（短边缩放 → 随机裁剪 → 水平翻转）在 dataloader 中
  在线完成，不需要离线特征缓存。
- **GPIC**（t2i）：[stanford-vision-lab/gpic](https://huggingface.co/datasets/stanford-vision-lab/gpic)，
  一个 WebDataset 目录（`{key}.json` 描述 + `{key}.jpg/.png`）。
  （t2i 训练器也支持通过 `--data-type=blip3o` 读取 BLIP3o 风格的 WebDataset 分片。）
- **COCO-2017 val**（t2i 在线监控）：GPIC t2i 训练器会在训练中记录
  COCO caption→image 的 FID + CLIPScore。把
  [BinLin203/COCO2017-Val](https://huggingface.co/datasets/BinLin203/COCO2017-Val)
  下载到本地目录，并把 `COCO_FID_PATH` 指向它（设 `COCO_FID_STEPS=0` 可跳过）。

把每个训练脚本顶部的 `DATA_DIR` 变量改成你本地的路径（脚本里默认是 `/path/to/...` 占位符）。

---

## 训练

GEAR 分三个阶段训练；每个脚本都把可选项以变量形式暴露在文件顶部
（编码器、模型尺寸、分词器、分辨率等）。所有脚本默认在单节点 8 卡上启动；多节点时设置
`NNODES` / `NODE_RANK` / `MASTER_ADDR` / `MASTER_PORT`，并在每个节点各启动一次。

| 阶段 | 脚本 | 作用 |
|---|---|---|
| 0 | [`scripts/train/train_tokenizer.sh`](scripts/train/train_tokenizer.sh) | 预热分词器以恢复其 GAN 判别器权重（VQ / LFQ / IBQ） |
| 1 | [`scripts/train/train_gear.sh`](scripts/train/train_gear.sh) | **GEAR 端到端**：用 REPA 联合训练分词器 + AR |
| 2（GEAR） | [`scripts/train/train_ar_gear.sh`](scripts/train/train_ar_gear.sh) | 冻结端到端分词器，继续把 AR 训得更久 |
| 2（基线） | [`scripts/train/train_ar_llamagen_repa.sh`](scripts/train/train_ar_llamagen_repa.sh) | 在冻结的预训练分词器上训练 LlamaGen-REPA 基线 |
| t2i（GPIC） | [`scripts/train/train_t2i_gpic.sh`](scripts/train/train_t2i_gpic.sh) | GPIC 文生图 |

```bash
# Stage 0 — 分词器（vq | lfq | ibq）
TOKENIZER=vq bash scripts/train/train_tokenizer.sh

# Stage 1 — GEAR 端到端（编码器/尺寸/分词器/温度/分辨率可选）
ENCODER=dinov2 SIZE=xl TOKENIZER=vq bash scripts/train/train_gear.sh

# Stage 2 — GEAR（继承 Stage-1 的 AR；或用 GEAR_MODE=vq-only 从零训 AR）
SIZE=xl GEAR_MODE=ar-init bash scripts/train/train_ar_gear.sh
#       — LlamaGen-REPA 基线
SIZE=xl bash scripts/train/train_ar_llamagen_repa.sh

# GPIC 文生图（RECIPE=gear|llamagen-repa）
RECIPE=gear bash scripts/train/train_t2i_gpic.sh
```

---

## 推理

GEAR 开源的是 **tokenizer**，不是 AR 生成器。要 *生成* 图像，请先用 [训练脚本](#训练) 在冻结的
GEAR tokenizer 上训练一个 AR，再用 `src/inference.py`（类别条件）或 `src/inference_t2i.py`
（文生图）采样。

若只想单独使用 tokenizer，encode → decode 即可重建图像：

```python
import torch, torchvision.transforms as T
from PIL import Image
from models import Tokenizers
from src.utils import load_pretrained_tokenizer_state_dict

# VQ-16；其他量化器用 "LFQ-16"（codebook_embed_dim=14）或 "IBQ-16"（codebook_embed_dim=256）
vq = Tokenizers["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
vq.load_state_dict(load_pretrained_tokenizer_state_dict("ckpts/GEAR-VQ/gear-vq.pt"), strict=False)
vq = vq.eval().cuda()

x = T.ToTensor()(Image.open("input.jpg").convert("RGB").resize((256, 256)))
x = (x * 2 - 1).unsqueeze(0).cuda()             # 归一化到 [-1, 1]
with torch.no_grad():
    recon, _ = vq(x)                             # encode -> quantize -> decode
out = (recon[0].clamp(-1, 1) + 1) / 2
T.ToPILImage()(out.cpu()).save("recon.png")
```

---

## 评测

一键驱动脚本在 [`scripts/eval/`](scripts/eval)。采样始终在 `gear` 环境中运行；每个 benchmark 的
**打分器** 有各自的环境（运行前请按链接的各 benchmark README 配置好）。

### ImageNet（类别条件）

| 指标 | 驱动脚本 | 打分环境 |
|---|---|---|
| gFID / sFID / IS / Precision / Recall | [`scripts/eval/eval_gfid.sh`](scripts/eval/eval_gfid.sh) | `adm`（ADM 评估器，TensorFlow） |
| 分词器重建（L1/PSNR/SSIM/rFID） | [`scripts/eval/eval_tokenizer.sh`](scripts/eval/eval_tokenizer.sh) | `gear` |

gFID 通路用一个独立的 `adm` 环境里的 ADM 评估器打分。先配置好它，并下载一次 ImageNet-256
参考批：

```bash
conda create -n adm python=3.10 -y
conda activate adm
pip install tensorflow==2.15.0 scipy requests tqdm numpy==1.23.5
pip install nvidia-pyindex
pip install nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12
# ImageNet-256 参考批（把 REF_NPZ 指向它）
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
```

然后在你自己训练的 AR 上跑 gFID（采样在 `gear`、打分在 `adm`；
改脚本顶部的路径，或在命令行内联传入）：

```bash
EXP_NAME=my-gear-xl \
CKPT=exps/<your-run>/checkpoints/0400000.pt \
AR_MODEL=LlamaGen-XL IMAGE_SIZE=256 \
OUT_DIR=infer_out REF_NPZ=/path/to/VIRTUAL_imagenet256_labeled.npz \
bash scripts/eval/eval_gfid.sh
```

> **关于分词器（`VQ_CKPT`）。** `train_gear.py`（端到端）产出的 checkpoint **已把分词器打包在内**
> （`vq` / `vq_ema`），此时把 `VQ_CKPT` 留空即可，脚本会直接从 checkpoint 读取
> （`eval_gfid.sh` 默认即为空）。而 `train_ar.py`（冻结分词器）产出的 checkpoint 需要通过
> `VQ_CKPT` 传入配套分词器（例如 `ckpts/GEAR-VQ/gear-vq.pt`）。

### 文生图

| Benchmark | 驱动脚本 | 打分环境 / README |
|---|---|---|
| GenEval | [`scripts/eval/eval_geneval.sh`](scripts/eval/eval_geneval.sh) | `geneval_eval` — [README](src/eval/geneval/README.md) |
| DPG-Bench | [`scripts/eval/eval_dpgbench.sh`](scripts/eval/eval_dpgbench.sh) | `dpgbench_eval` — [README](src/eval/dpgbench/README.md) |
| WISE（**WISE_Verified**） | [`scripts/eval/eval_wise.sh`](scripts/eval/eval_wise.sh) | 在 `gear` 中用 vLLM judge（`pip install vllm`）— [README](src/eval/wise/README.md) |
| GPIC（FD-DINOv2） | [`scripts/eval/eval_gpic.sh`](scripts/eval/eval_gpic.sh) | `gear` — [README](src/eval/gpic/README.md) |
| COCO-FID + CLIPScore | [`scripts/eval/eval_coco.sh`](scripts/eval/eval_coco.sh) | `gear`（单次推理）— [README](src/eval/coco/README.md) |

把每个 t2i 驱动脚本指向你用 `train_t2i_gpic.sh` 训练得到的模型（`--ar-model LlamaGen-1B`，
`IMAGE_SIZE=256`），并通过 `--vq-ckpt-path` 传入其分词器（GEAR 用 `gear-vq.pt`，warm-up 基线用
`vq-with-gan.pt`）。这里的 WISE 是 **WISE_Verified** 协议（刷新后的 prompt 集 + 本地 Qwen3.5
judge），**不是** 旧的 GPT-4o 版本。

---

## 引用

```bibtex
@article{gear,
  title   = {GEAR: Guided End-to-End AutoRegression for Image Synthesis},
  author  = {GEAR authors},
  year    = {2026}
}

@article{ifsq_llamagenrepa,
  title   = {iFSQ: Improving FSQ for Image Generation with 1 Line of Code},
  author  = {Lin, Bin and Li, Zongjian and Niu, Yuwei and Gong, Kaixiong and
             Ge, Yunyang and Lin, Yunlong and Zheng, Mingzhe and Zhang, JianWei and
             Yang, Miles and Zhong, Zhao and others},
  journal = {arXiv preprint arXiv:2601.17124},
  year    = {2026}
}
```

## 致谢

GEAR 基于 [LlamaGen](https://github.com/FoundationVision/LlamaGen)、
[REPA](https://github.com/sihyun-yu/REPA) / [REPA-E](https://github.com/End2End-Diffusion/REPA-E)、
[Open-MAGVIT2](https://github.com/TencentARC/SEED-Voken) 与
[IBQ](https://github.com/TencentARC/SEED-Voken) 构建。LlamaGen-REPA 基线参考了
[iFSQ / LlamaGen-REPA](https://github.com/Tencent-Hunyuan/iFSQ)，评测框架
（GenEval / DPG-Bench / WISE 集成）改编自
[UniWorld-V1](https://github.com/PKU-YuanGroup/UniWorld)，并基于原始的
[GenEval](https://github.com/djghosh13/geneval)、
[DPG-Bench](https://github.com/TencentQQGYLab/ELLA) 与
[WISE](https://github.com/PKU-YuanGroup/WISE)。文生图实验使用了
[GPIC](https://github.com/keshik6/gpic) 数据集及其评测工具。在此一并致谢。
