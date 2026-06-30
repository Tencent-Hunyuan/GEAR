# COCO-FID + CLIPScore (src)

COCO caption→image evaluation for `src` t2i checkpoints. Reports
**FID** (generated vs COCO ground-truth images) and **CLIPScore** (image-text
alignment), both computed over the **same** COCO captions in **one pass**.
Everything runs in the single `gear` env — no extra benchmark environment, and
**no PNGs are written** (images are scored in memory).

| step | file | env | output |
|------|------|-----|--------|
| 1+2. sample + score | `coco_fid_eval.py` | `gear` | FID + CLIPScore (`<out>/<exp>/<step>/<tag>/coco_fid.{json,txt}`) |

Unlike GenEval / DPG-Bench (sample → store PNGs → score in a separate env), the
COCO evaluator drives sampling through a `caption → PIL` callback and computes
both metrics inline, so it is a single command.

### Get the reference data

A directory of COCO-val parquet files with `caption` + `image` columns. Download
[BinLin203/COCO2017-Val](https://huggingface.co/datasets/BinLin203/COCO2017-Val):

```bash
huggingface-cli download BinLin203/COCO2017-Val \
    --repo-type dataset --local-dir coco_data
# -> point COCO_DATASET_PATH at the dir holding the .parquet files
```

`--num-samples` is auto-capped to the available count (COCO val is 5000 images;
the standard COCO-30k protocol needs a 30k-caption parquet set instead).

## One-shot (recommended)

Edit the paths at the top of [`scripts/eval/eval_coco.sh`](../../../scripts/eval/eval_coco.sh)
(`CKPT`, `VQ_CKPT`, `OUTPUT_DIR`, `COCO_DATASET_PATH`) then run it. Architecture
knobs are env-overridable:

```bash
AR_MODEL=LlamaGen-1B IMAGE_SIZE=256 TEXT_ENCODER=Qwen/Qwen3-1.7B CFG_SCALE=1.5 \
bash scripts/eval/eval_coco.sh
```

For published HF weights set `VQ_CKPT` to the paired tokenizer
(`Warmup-VQ/vq-with-gan.pt` for LlamaGen-REPA, `GEAR-VQ/gear-vq.pt` for GEAR).

## Manual (gear env)

Published checkpoints have their `args` stripped, so pass `--ar-model` /
`--image-size` / `--text-encoder` explicitly:

```bash
accelerate launch --num_processes 8 \
    src/eval/coco/coco_fid_eval.py \
    --ckpt-path          /path/to/gpic.pt \
    --vq-ckpt-path       /path/to/vq.pt \
    --ar-model           LlamaGen-1B --image-size 256 \
    --text-encoder       Qwen/Qwen3-1.7B \
    --coco-dataset-path  /path/to/COCO2017-Val \
    --clip-model-path    openai/clip-vit-base-patch32 \
    --num-samples 30000 --coco-image-size 256 \
    --cfg-scale 1.5 \
    --out-dir /path/to/coco_fid_out
```

The FID + CLIPScore print to stdout and are saved under
`<out-dir>/<exp>/<step>/cfg<g>-temp<g>/coco_fid.{json,txt}`.
