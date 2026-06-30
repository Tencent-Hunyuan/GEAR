#!/bin/bash
# ============================================================================
# GPIC FD-DINOv2 evaluation for a Stage-2 t2i checkpoint.
#
#   1. Sample one image per prompt (src AR + frozen VQ) -> samples.npz.
#   2. Score FD-DINOv2 + PRDC + MMD against the GPIC test reference stats
#      (src/eval/gpic/gpic_eval_dino.py, runs in the same env).
#
# Stage-2 t2i ckpts store ONLY the AR, so the frozen tokenizer is supplied via
# VQ_CKPT. GPIC is evaluated without CFG (cfg=1).
# ============================================================================

set -e
cd "$(dirname "$0")/../.."   # repo root

# ---- what to evaluate (edit these, or override inline: CKPT=... bash ...) ---
EXP_NAME="${EXP_NAME:-my-run}"                                       # run name (= <run> in CKPT)
CKPT="${CKPT:-/path/to/exps/${EXP_NAME}/checkpoints/0100000.pt}"     # AR checkpoint
# Frozen tokenizer: Stage-0 ckpt (llamagen-repa) or Stage-1 e2e ckpt (gear).
VQ_CKPT="${VQ_CKPT:-/path/to/checkpoints/0400000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/gpic_infer_out}"
PROMPTS_JSONL="${PROMPTS_JSONL:-src/eval/gpic/eval_prompts/gpic_eval_50k.jsonl}"  # in-repo prompt set
TEST_STATS="${TEST_STATS:-/path/to/gpic/reference_stats/test_stats.npz}"

# Model architecture (REQUIRED for published HF weights; their args snapshot is
# stripped on upload). IMAGE_SIZE = the resolution the AR was trained at.
AR_MODEL="${AR_MODEL:-LlamaGen-1B}"           # gpic t2i model
IMAGE_SIZE="${IMAGE_SIZE:-256}"               # 256 | 512
TEXT_ENCODER="${TEXT_ENCODER:-Qwen/Qwen3-1.7B}"  # local dir or HF id
TEXT_MAX_LEN="${TEXT_MAX_LEN:-300}"

CFG_SCALE="${CFG_SCALE:-1}"
NUM_PROC="${NUM_PROC:-8}"
PER_PROC_BS="${PER_PROC_BS:-32}"

# ---- 1. sample -> samples.npz ----------------------------------------------
accelerate launch \
    --main_process_ip 127.0.0.1 --main_process_port 12345 \
    --num_processes "${NUM_PROC}" --num_machines 1 \
  src/inference_t2i.py \
    --ckpt-path "${CKPT}" \
    --vq-ckpt-path "${VQ_CKPT}" \
    --ar-model "${AR_MODEL}" \
    --image-size "${IMAGE_SIZE}" \
    --text-encoder "${TEXT_ENCODER}" \
    --text-max-len "${TEXT_MAX_LEN}" \
    --exp-name "${EXP_NAME}" \
    --prompts-jsonl "${PROMPTS_JSONL}" \
    --output-dir "${OUTPUT_DIR}" \
    --per-proc-batch-size "${PER_PROC_BS}" \
    --cfg-scale "${CFG_SCALE}"

# ---- 2. FD-DINOv2 + PRDC + MMD ---------------------------------------------
# inference_t2i.py writes <OUTPUT_DIR>/<exp>/step<step>/cfg<g>-temp<g>/samples.npz.
STEP_TAG="step$(basename "${CKPT}" .pt)"
SAMPLING_TAG="$(printf 'cfg%g-temp%g' "${CFG_SCALE}" 1.0)"
RUN_DIR="${OUTPUT_DIR}/${EXP_NAME}/${STEP_TAG}/${SAMPLING_TAG}"
python src/eval/gpic/gpic_eval_dino.py \
    "${RUN_DIR}/samples.npz" \
    "${TEST_STATS}" \
    --metrics fd,prdc,mmd \
    --out-dir "${RUN_DIR}"
