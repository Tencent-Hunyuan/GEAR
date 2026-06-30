#!/bin/bash
# ============================================================================
# WISE_Verified text-to-image evaluation for a Stage-2 t2i checkpoint.
#
#   1. Sample one image per prompt (src AR + frozen VQ).
#   2. Judge each image with Qwen3.5-35B-A3B served by a local vLLM
#      OpenAI-compatible server (launched here in FAST_START mode).
#   3. Aggregate the binary WiScore.
#
# Stage-2 t2i ckpts store ONLY the AR, so the frozen tokenizer is supplied via
# VQ_CKPT. The vLLM judge is launched on a single GPU (tp=1) in FAST_START mode
# (Triton MoE+GDN, autotune off, enforce-eager -> server up in ~2-3 min, no
# cutlass JIT). Sampling knobs other than CFG use the step1 defaults.
# ============================================================================

set -e
cd "$(dirname "$0")/../.."   # repo root

# ---- what to evaluate (edit these, or override inline: CKPT=... bash ...) ---
EXP_NAME="${EXP_NAME:-my-run}"                                       # run name (= <run> in CKPT)
CKPT="${CKPT:-/path/to/exps/${EXP_NAME}/checkpoints/0100000.pt}"     # AR checkpoint
# Frozen tokenizer: Stage-0 ckpt (llamagen-repa) or Stage-1 e2e ckpt (gear).
VQ_CKPT="${VQ_CKPT:-/path/to/checkpoints/0400000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/wise_out/${EXP_NAME}}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-Qwen/Qwen3.5-35B-A3B}"           # judge model (HF id or local dir)

# Model architecture (REQUIRED for published HF weights; their args snapshot is
# stripped on upload). IMAGE_SIZE = the resolution the AR was trained at.
AR_MODEL="${AR_MODEL:-LlamaGen-1B}"           # LlamaGen-1B
IMAGE_SIZE="${IMAGE_SIZE:-256}"               # 256 | 512
TEXT_ENCODER="${TEXT_ENCODER:-Qwen/Qwen3-1.7B}"  # local dir or HF id
TEXT_MAX_LEN="${TEXT_MAX_LEN:-300}"

CFG_SCALE="${CFG_SCALE:-4.0}"
NUM_PROC="${NUM_PROC:-8}"
PER_PROC_BS="${PER_PROC_BS:-32}"

# In-repo verified prompt set (shipped with the repo).
DATA_DIR="src/eval/wise/data_verified"

# vLLM judge server (single GPU, tp=1).
JUDGE_MODEL="Qwen3.5-35B-A3B"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_API_BASE="http://127.0.0.1:${VLLM_PORT}/v1"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
SERVE_TIMEOUT="${SERVE_TIMEOUT:-1800}"
MAX_WORKERS="${MAX_WORKERS:-96}"

IMAGE_DIR="${OUTPUT_DIR}/samples"
RESULTS_DIR="${OUTPUT_DIR}/Results-qwen35"
SUMMARY_TXT="${RESULTS_DIR}/wise_summary.txt"
mkdir -p "${RESULTS_DIR}"

# Local endpoint must bypass any proxy.
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

# ---- 1. sample images ------------------------------------------------------
accelerate launch \
    --main_process_ip 127.0.0.1 --main_process_port 12347 \
    --num_processes "${NUM_PROC}" --num_machines 1 \
  src/eval/wise/step1_gen_samples.py \
    --ckpt-path "${CKPT}" \
    --vq-ckpt-path "${VQ_CKPT}" \
    --ar-model "${AR_MODEL}" \
    --image-size "${IMAGE_SIZE}" \
    --text-encoder "${TEXT_ENCODER}" \
    --text-max-len "${TEXT_MAX_LEN}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${IMAGE_DIR}" \
    --per-proc-batch-size "${PER_PROC_BS}" \
    --cfg-scale "${CFG_SCALE}"

# ---- 2. launch the Qwen vLLM judge (FAST_START, single GPU) ----------------
VLLM_PID=""
cleanup_vllm() { [[ -n "${VLLM_PID}" ]] && kill "${VLLM_PID}" 2>/dev/null || true; }
trap cleanup_vllm EXIT

vllm serve "${QWEN_MODEL_PATH}" \
    --served-model-name "${JUDGE_MODEL}" \
    --host 0.0.0.0 --port "${VLLM_PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --moe-backend triton --gdn-prefill-backend triton \
    --no-enable-flashinfer-autotune --enforce-eager \
    > "${OUTPUT_DIR}/vllm_serve.log" 2>&1 &
VLLM_PID=$!

echo "[wise] waiting up to ${SERVE_TIMEOUT}s for vLLM (log: ${OUTPUT_DIR}/vllm_serve.log)"
waited=0
until curl -sf "${VLLM_API_BASE}/models" >/dev/null 2>&1; do
    sleep 10; waited=$((waited + 10))
    kill -0 "${VLLM_PID}" 2>/dev/null || { echo "[wise] vLLM exited early; see vllm_serve.log" >&2; exit 1; }
    (( waited >= SERVE_TIMEOUT )) && { echo "[wise] vLLM not ready after ${SERVE_TIMEOUT}s" >&2; exit 1; }
done
echo "[wise] vLLM endpoint is up."

# ---- 2b. judge each category -----------------------------------------------
judge() {  # $1=verified json, $2=name
    python src/eval/wise/step2_vllm_eval.py \
        --json_path "${DATA_DIR}/$1" \
        --output_dir "${RESULTS_DIR}/$2" \
        --image_dir "${IMAGE_DIR}" \
        --api_key EMPTY --api_base "${VLLM_API_BASE}" --model "${JUDGE_MODEL}" \
        --result_full "${RESULTS_DIR}/$2_full_results.json" \
        --result_scores "${RESULTS_DIR}/$2_scores_results.jsonl" \
        --max_workers "${MAX_WORKERS}"
}
judge cultural_common_sense_verified.json     cultural_common_sense
judge spatio-temporal_reasoning_verified.json spatio-temporal_reasoning
judge natural_science_verified.json           natural_science

# ---- 3. aggregate the binary WiScore ---------------------------------------
python src/eval/wise/step3_calculate.py \
    "${RESULTS_DIR}/cultural_common_sense_scores_results.jsonl" \
    "${RESULTS_DIR}/natural_science_scores_results.jsonl" \
    "${RESULTS_DIR}/spatio-temporal_reasoning_scores_results.jsonl" \
    --category all | tee "${SUMMARY_TXT}"
