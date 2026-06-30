# DPG-Bench (src)

DPG-Bench for `src` t2i checkpoints. The metric code (`step2`) is
the original [DPG-Bench](https://github.com/TencentQQGYLab/ELLA) ported via
UniWorld and is **unchanged** — it only consumes generated images. Only `step1`
was rewritten to sample from our AR+VQ model (shared with the GPIC eval through
`src/eval/_common.py`, which imports the loaders from
`src/inference_t2i.py`).

DPG-Bench scores a **2x2 grid** per prompt, so step-1 samples `pic_num` (=4)
images, tiles them, and step-2 crops each tile back out with `--resolution`
(which must equal step-1's `--resize`).

| step | file | env | output |
|------|------|-----|--------|
| 1. generate | `step1_gen_samples.py` | `gear` | `<run>/<key>.png` (2x2 grids) |
| 2. score+summary | `step2_compute_dpg_bench.py` | `dpgbench_eval` | `<run-parent>/dpgbench_<tag>.txt` |

## Setup (do this first)

Sampling (step 1) runs in the `gear` env. The scorer (step 2) needs a separate
`dpgbench_eval` env plus the mPLUG VQA model. Build it once:

```bash
conda create -n dpgbench_eval python=3.10 -y
conda activate dpgbench_eval

# scorer deps (incl. modelscope; the main `gear` env does NOT have it)
cd src/eval/dpgbench
pip install -r requirements.txt

# mPLUG VQA model (download once, from THIS dpgbench_eval env -- modelscope
# lives here, not in `gear`)
modelscope download --model 'iic/mplug_visual-question-answering_coco_large_en' \
  --local_dir /path/to/mplug
cd -
```

Point the script's `MPLUG_PATH` at the downloaded model.

## One-shot (recommended)

Edit the paths at the top of [`scripts/eval/eval_dpgbench.sh`](../../../scripts/eval/eval_dpgbench.sh)
(`CKPT`, `VQ_CKPT`, `OUTPUT_DIR`, `MPLUG_PATH`) then run it. It samples in the `gear`
env and scores in the `dpgbench_eval` env. Architecture knobs are env-overridable:

```bash
AR_MODEL=LlamaGen-XL IMAGE_SIZE=512 TEXT_ENCODER=Qwen/Qwen3-1.7B CFG_SCALE=4.0 \
bash scripts/eval/eval_dpgbench.sh
```

For published HF weights set `VQ_CKPT` to the paired tokenizer (e.g.
`Warmup-VQ/vq-with-gan.pt` or `GEAR-VQ/gear-vq.pt`).

## Manual, step by step

### Step 1 — generate (gear env)

Published checkpoints have their `args` stripped, so pass `--ar-model` /
`--image-size` / `--text-encoder` explicitly:

```bash
accelerate launch --num_processes 8 \
    src/eval/dpgbench/step1_gen_samples.py \
    --ckpt-path     /path/to/blip3o-512-ft.pt \
    --vq-ckpt-path  /path/to/vq.pt \
    --ar-model      LlamaGen-XL --image-size 512 \
    --text-encoder  Qwen/Qwen3-1.7B \
    --prompts-json  src/eval/dpgbench/eval_prompts/dpgbench_prompts.json \
    --output-dir    /path/to/dpgbench_out/run \
    --pic-num 4 --resize 512 --cfg-scale 1.75
```

### Step 2 — mPLUG VQA scoring + summary (dpgbench_eval env)

Uses the env + model from [Setup](#setup-do-this-first). `--resolution` must equal
step-1's `--resize` (512), and keep `--res_path` **outside** `--image_root_path`
(step-2 treats every file in that dir as a generated image):

```bash
conda activate dpgbench_eval
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --num_machines 1 --num_processes 8 --multi_gpu --mixed_precision fp16 \
    src/eval/dpgbench/step2_compute_dpg_bench.py \
    --image_root_path /path/to/dpgbench_out/run \
    --resolution 512 \
    --pic_num 4 \
    --res_path /path/to/dpgbench_out/run_dpgbench.txt \
    --vqa_model mplug \
    --mplug_local_path /path/to/mplug \
    --csv src/eval/dpgbench/eval_prompts/dpgbench.csv
cat /path/to/dpgbench_out/run_dpgbench.txt
```
