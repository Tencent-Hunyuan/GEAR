#!/bin/bash
# ============================================================================
# GenEval text-to-image evaluation for a Stage-2 t2i checkpoint.
#
#   1. Sample images per prompt (src AR + frozen VQ).
#   2. Mask2Former + CLIP object/attribute checker -> geneval.jsonl
#   3. Summarise per-task accuracy -> geneval.txt
# Stages 2-3 run in the `geneval_eval` conda env (mmdet/mmcv/open_clip).
#
# Stage-2 t2i ckpts store ONLY the AR, so the frozen tokenizer is supplied via
# VQ_CKPT. Sampling knobs other than CFG use the step1 defaults. To score the
# rewritten long prompts instead, point PROMPTS_JSONL at *_long.jsonl.
# ============================================================================

set -e
cd "$(dirname "$0")/../.."   # repo root

# ---- what to evaluate (edit these, or override inline: CKPT=... bash ...) ---
EXP_NAME="${EXP_NAME:-my-run}"                                       # run name (= <run> in CKPT)
CKPT="${CKPT:-/path/to/exps/${EXP_NAME}/checkpoints/0100000.pt}"     # AR checkpoint
# Frozen tokenizer: Stage-0 ckpt (llamagen-repa) or Stage-1 e2e ckpt (gear).
VQ_CKPT="${VQ_CKPT:-/path/to/checkpoints/0400000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/geneval_out/${EXP_NAME}}"
DETECTOR_PATH="${DETECTOR_PATH:-/path/to/geneval_cache/detector}"    # Mask2Former weights
CACHE_DIR="${CACHE_DIR:-/path/to/geneval_cache/cache_dir}"           # open_clip cache
GENEVAL_ENV="${GENEVAL_ENV:-geneval_eval}"                           # conda env for the scorer

# Model architecture (REQUIRED for published HF weights; their args snapshot is
# stripped on upload). IMAGE_SIZE = the resolution the AR was trained at.
AR_MODEL="${AR_MODEL:-LlamaGen-1B}"           # LlamaGen-1B
IMAGE_SIZE="${IMAGE_SIZE:-256}"               # 256 | 512
TEXT_ENCODER="${TEXT_ENCODER:-Qwen/Qwen3-1.7B}"  # local dir or HF id
TEXT_MAX_LEN="${TEXT_MAX_LEN:-300}"

CFG_SCALE="${CFG_SCALE:-4.0}"
NUM_PROC="${NUM_PROC:-8}"
PER_PROC_BS="${PER_PROC_BS:-32}"

# In-repo prompt set (use evaluation_metadata_long.jsonl for the rewritten set).
PROMPTS_JSONL="src/eval/geneval/eval_prompts/evaluation_metadata.jsonl"

IMAGE_DIR="${OUTPUT_DIR}/samples"
RESULT_JSONL="${OUTPUT_DIR}/geneval.jsonl"
RESULT_TXT="${OUTPUT_DIR}/geneval.txt"

# ---- 1. sample images ------------------------------------------------------
accelerate launch \
    --main_process_ip 127.0.0.1 --main_process_port 12345 \
    --num_processes "${NUM_PROC}" --num_machines 1 \
  src/eval/geneval/step1_gen_samples.py \
    --ckpt-path "${CKPT}" \
    --vq-ckpt-path "${VQ_CKPT}" \
    --ar-model "${AR_MODEL}" \
    --image-size "${IMAGE_SIZE}" \
    --text-encoder "${TEXT_ENCODER}" \
    --text-max-len "${TEXT_MAX_LEN}" \
    --prompts-jsonl "${PROMPTS_JSONL}" \
    --output-dir "${IMAGE_DIR}" \
    --per-proc-batch-size "${PER_PROC_BS}" \
    --cfg-scale "${CFG_SCALE}"

# ---- 2-3. detection checker + summary (geneval_eval env) -------------------
conda activate "${GENEVAL_ENV}"
CUDA_VISIBLE_DEVICES=0 CACHE_DIR="${CACHE_DIR}" \
  python src/eval/geneval/step2_run_geneval.py \
    "${IMAGE_DIR}" \
    --outfile "${RESULT_JSONL}" \
    --model-path "${DETECTOR_PATH}"
python src/eval/geneval/step3_summary_score.py "${RESULT_JSONL}" > "${RESULT_TXT}"
cat "${RESULT_TXT}"
