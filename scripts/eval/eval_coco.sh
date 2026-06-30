#!/bin/bash
# ============================================================================
# COCO caption->image FID + CLIPScore for a Stage-2 t2i checkpoint.
#
# Single stage (no separate scorer env): src/eval/coco/coco_fid_eval.py
# drives sampling through a caption->PIL callback and computes BOTH FID
# (vs COCO ground-truth) and CLIPScore in one pass over the same COCO data,
# all in the `gear` env. Results print to stdout and are saved to
# <OUTPUT_DIR>/<exp>/<step>/<cfg-tag>/coco_fid.{json,txt}.
#
# Stage-2 t2i ckpts store ONLY the AR (the VQ is frozen during training), so
# the tokenizer must be supplied via VQ_CKPT.
# ============================================================================

set -e
cd "$(dirname "$0")/../.."   # repo root

# ---- what to evaluate (edit these, or override inline: CKPT=... bash ...) ---
EXP_NAME="${EXP_NAME:-my-run}"                                       # run name (= <run> in CKPT)
CKPT="${CKPT:-/path/to/exps/${EXP_NAME}/checkpoints/0390625.pt}"     # AR checkpoint
# Frozen tokenizer: Stage-0 ckpt (llamagen-repa) or Stage-1 e2e ckpt (gear).
VQ_CKPT="${VQ_CKPT:-/path/to/checkpoints/0400000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/coco_fid_out}"

# COCO val reference: a dir of parquet files (columns: caption, image).
# Download BinLin203/COCO2017-Val from the Hub and point COCO_DATASET_PATH at it.
COCO_DATASET_PATH="${COCO_DATASET_PATH:-/path/to/COCO2017-Val}"
CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-base-patch32}"             # local dir or HF id (empty disables CLIPScore)

# Model architecture (REQUIRED for published HF weights; their args snapshot is
# stripped on upload). IMAGE_SIZE = the resolution the AR was trained at.
AR_MODEL="${AR_MODEL:-LlamaGen-1B}"           # LlamaGen-1B
IMAGE_SIZE="${IMAGE_SIZE:-256}"               # 256 | 512
TEXT_ENCODER="${TEXT_ENCODER:-Qwen/Qwen3-1.7B}"  # local dir or HF id
TEXT_MAX_LEN="${TEXT_MAX_LEN:-300}"

CFG_SCALE="${CFG_SCALE:-1.5}"
NUM_PROC="${NUM_PROC:-8}"
INCEPTION_BS="${INCEPTION_BS:-32}"
NUM_SAMPLES="${NUM_SAMPLES:-30000}"           # COCO captions to score (auto-capped to the dataset size)
COCO_IMAGE_SIZE="${COCO_IMAGE_SIZE:-256}"     # square size for FID (conventionally 256)

# ---- sample + score (FID + CLIPScore in one pass) --------------------------
accelerate launch \
    --main_process_ip 127.0.0.1 --main_process_port 12349 \
    --num_processes "${NUM_PROC}" --num_machines 1 \
  src/eval/coco/coco_fid_eval.py \
    --ckpt-path "${CKPT}" \
    --vq-ckpt-path "${VQ_CKPT}" \
    --ar-model "${AR_MODEL}" \
    --image-size "${IMAGE_SIZE}" \
    --text-encoder "${TEXT_ENCODER}" \
    --text-max-len "${TEXT_MAX_LEN}" \
    --exp-name "${EXP_NAME}" \
    --coco-dataset-path "${COCO_DATASET_PATH}" \
    --num-samples "${NUM_SAMPLES}" \
    --coco-image-size "${COCO_IMAGE_SIZE}" \
    --inception-batch-size "${INCEPTION_BS}" \
    --clip-model-path "${CLIP_MODEL}" \
    --cfg-scale "${CFG_SCALE}" \
    --out-dir "${OUTPUT_DIR}"
