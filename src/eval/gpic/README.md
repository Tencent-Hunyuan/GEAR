# GPIC FD-DINOv2 (src)

GPIC text-to-image evaluation for `src` t2i checkpoints. Unlike the
prompt-following benchmarks (GenEval / DPG-Bench / WISE), GPIC measures
**distributional fidelity** of generated images against a held-out reference set:
**FD-DINOv2** plus PRDC (precision/recall/density/coverage) and MMD. Everything runs
in the single `gear` env — no extra benchmark environment.

| step | file | env | output |
|------|------|-----|--------|
| 1. generate | `src/inference_t2i.py` | `gear` | `<out>/<exp>/<step>/<tag>/samples.npz` |
| 2. score    | `gpic_eval_dino.py`        | `gear` | FD-DINOv2 / PRDC / MMD |

The prompt list (`eval_prompts/gpic_eval_50k.jsonl`) is matched to the reference
stats (`reference_stats/test_stats.npz`, one image per reference key, generated from
that key's own caption).

### Get the reference stats

`test_stats.npz` lives in the (gated) GPIC dataset:
[stanford-vision-lab/gpic › reference_stats](https://huggingface.co/datasets/stanford-vision-lab/gpic/tree/main/reference_stats).
Accept the dataset conditions on the Hub, then download it (auth required):

```bash
huggingface-cli download stanford-vision-lab/gpic \
    reference_stats/test_stats.npz \
    --repo-type dataset --local-dir gpic_data
# -> gpic_data/reference_stats/test_stats.npz   (point TEST_STATS at this)
```

## One-shot (recommended)

Edit the paths at the top of [`scripts/eval/eval_gpic.sh`](../../../scripts/eval/eval_gpic.sh)
(`CKPT`, `VQ_CKPT`, `OUTPUT_DIR`, `TEST_STATS`) then run it. GPIC is evaluated
**without CFG** (`cfg=1`). Architecture knobs are env-overridable:

```bash
AR_MODEL=LlamaGen-1B IMAGE_SIZE=256 TEXT_ENCODER=Qwen/Qwen3-1.7B \
bash scripts/eval/eval_gpic.sh
```

For published HF weights set `VQ_CKPT` to the paired tokenizer
(`Warmup-VQ/vq-with-gan.pt` for LlamaGen-REPA, `GEAR-VQ/gear-vq.pt` for GEAR).

## Manual, step by step

### Step 1 — generate (gear env)

Published checkpoints have their `args` stripped, so pass `--ar-model` /
`--image-size` / `--text-encoder` explicitly:

```bash
accelerate launch --num_processes 8 \
    src/inference_t2i.py \
    --ckpt-path     /path/to/gpic.pt \
    --vq-ckpt-path  /path/to/vq.pt \
    --ar-model      LlamaGen-1B --image-size 256 \
    --text-encoder  Qwen/Qwen3-1.7B \
    --prompts-jsonl src/eval/gpic/eval_prompts/gpic_eval_50k.jsonl \
    --output-dir    /path/to/gpic_out \
    --per-proc-batch-size 32 --cfg-scale 1
```

### Step 2 — FD-DINOv2 + PRDC + MMD

```bash
python src/eval/gpic/gpic_eval_dino.py \
    /path/to/gpic_out/<exp>/<step>/<tag>/samples.npz \
    /path/to/gpic/reference_stats/test_stats.npz \
    --metrics fd,prdc,mmd
```
