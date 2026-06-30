#!/bin/bash
# ============================================================================
# DPG-Bench text-to-image evaluation for a Stage-2 t2i checkpoint.
#
#   1. Sample a 2x2 grid per prompt (src AR + frozen VQ).
#   2. mPLUG VQA scoring, in the `dpgbench_eval` conda env.
#
# Stage-2 t2i ckpts store ONLY the AR (the VQ is frozen during training), so
# the tokenizer must be supplied via VQ_CKPT. All sampling knobs other than CFG
# (temperature / top-k / top-p / pic-num / resize) are the defaults baked into
# src/eval/dpgbench/step1_gen_samples.py.
# ============================================================================

set -e
cd "$(dirname "$0")/../.."   # repo root

# ---- what to evaluate (edit these, or override inline: CKPT=... bash ...) ---
EXP_NAME="${EXP_NAME:-my-run}"                                       # run name (= <run> in CKPT)
CKPT="${CKPT:-/path/to/exps/${EXP_NAME}/checkpoints/0100000.pt}"     # AR checkpoint
# Frozen tokenizer: Stage-0 ckpt (llamagen-repa) or Stage-1 e2e ckpt (gear).
VQ_CKPT="${VQ_CKPT:-/path/to/checkpoints/0400000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/dpgbench_out/${EXP_NAME}}"
MPLUG_PATH="${MPLUG_PATH:-/path/to/iic/mplug_visual-question-answering_coco_large_en}"
DPG_ENV="${DPG_ENV:-dpgbench_eval}"                                  # conda env for the mPLUG scorer

# Model architecture (REQUIRED for published HF weights; their args snapshot is
# stripped on upload). IMAGE_SIZE = the resolution the AR was trained at.
AR_MODEL="${AR_MODEL:-LlamaGen-1B}"           # LlamaGen-1B
IMAGE_SIZE="${IMAGE_SIZE:-256}"               # 256 | 512
TEXT_ENCODER="${TEXT_ENCODER:-Qwen/Qwen3-1.7B}"  # local dir or HF id
TEXT_MAX_LEN="${TEXT_MAX_LEN:-300}"

CFG_SCALE="${CFG_SCALE:-4.0}"
NUM_PROC="${NUM_PROC:-8}"
PER_PROC_BS="${PER_PROC_BS:-32}"

# In-repo prompt set (shipped with the repo).
PROMPTS_JSON="src/eval/dpgbench/eval_prompts/dpgbench_prompts.json"
DPG_CSV="src/eval/dpgbench/eval_prompts/dpgbench.csv"

IMAGE_DIR="${OUTPUT_DIR}/samples"
RESULT_TXT="${OUTPUT_DIR}/dpgbench.txt"

# ---- 1. sample 2x2 grids ---------------------------------------------------
accelerate launch \
    --main_process_ip 127.0.0.1 --main_process_port 12346 \
    --num_processes "${NUM_PROC}" --num_machines 1 \
  src/eval/dpgbench/step1_gen_samples.py \
    --ckpt-path "${CKPT}" \
    --vq-ckpt-path "${VQ_CKPT}" \
    --ar-model "${AR_MODEL}" \
    --image-size "${IMAGE_SIZE}" \
    --text-encoder "${TEXT_ENCODER}" \
    --text-max-len "${TEXT_MAX_LEN}" \
    --prompts-json "${PROMPTS_JSON}" \
    --output-dir "${IMAGE_DIR}" \
    --per-proc-batch-size "${PER_PROC_BS}" \
    --cfg-scale "${CFG_SCALE}"

# ---- 2. mPLUG VQA scoring (dpgbench_eval env) ------------------------------
# --resolution / --pic_num must match step1's defaults (512 / 4).
conda activate "${DPG_ENV}"
accelerate launch --num_machines 1 --num_processes "${NUM_PROC}" \
    --multi_gpu --mixed_precision fp16 \
  src/eval/dpgbench/step2_compute_dpg_bench.py \
    --image_root_path "${IMAGE_DIR}" \
    --resolution 512 \
    --pic_num 4 \
    --res_path "${RESULT_TXT}" \
    --vqa_model mplug \
    --mplug_local_path "${MPLUG_PATH}" \
    --csv "${DPG_CSV}"
cat "${RESULT_TXT}"
