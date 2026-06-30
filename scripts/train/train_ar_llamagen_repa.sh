#!/bin/bash
# ============================================================================
# Stage 2 -- LlamaGen-REPA baseline.
#
# Trains a LlamaGen AR + REPA on top of a FROZEN Stage-0 tokenizer. This is
# the plain (non-e2e) baseline: the tokenizer is the one produced by
# scripts/train_tokenizer.sh, NOT an e2e-tuned Stage-1 tokenizer. No AR warm
# start, so the AR trains for the full 4M steps.
#
# (The e2e flavour -- our "GEAR" recipe -- lives in
#  scripts/train_ar_gear.sh.)
#
# Fixed for this script: REPA teacher = dinov2, tokenizer = vq. Only the AR
# model size is selectable.
#
# Single node, 8 GPUs. For multi-node set NNODES / NODE_RANK / MASTER_ADDR /
# MASTER_PORT and launch once per node.
# ============================================================================

set -e
cd "$(dirname "$0")/../.."   # repo root, so src/... paths resolve

# ---- choose the AR size (default: xl) --------------------------------------
SIZE="${SIZE:-xl}"        # b | l | xl   (AR model + REPA tap depth 4 / 8 / 12)

# ---- paths (edit these) ----------------------------------------------------
DATA_DIR="/path/to/imagenet/train"
# Frozen tokenizer = a Stage-0 checkpoint (output of scripts/train_tokenizer.sh).
# Loaded EMA-first (--vq-use-ema is on by default in train_ar.py).
VQ_CKPT="/path/to/exps/vq16-stage0-bs256-400k/checkpoints/0400000.pt"

# ---- optional eval (set EVAL_STEPS=0 to skip; needs FID_REF) ----------------
FID_REF="/path/to/VIRTUAL_imagenet256_labeled.npz"   # AR-generation FID ref stats
EVAL_STEPS="${EVAL_STEPS:-50000}"                    # AR FID cadence (0 disables)

# ---- launch ----------------------------------------------------------------
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ===========================================================================
# SIZE -> --ar-model + --encoder-depth (REPA tap layer; matches Stage 1)
# ===========================================================================
case "${SIZE}" in
    b)  AR_MODEL="LlamaGen-B";  ENCODER_DEPTH=4  ;;
    l)  AR_MODEL="LlamaGen-L";  ENCODER_DEPTH=8  ;;
    xl) AR_MODEL="LlamaGen-XL"; ENCODER_DEPTH=12 ;;
    *) echo "Unknown SIZE='${SIZE}' (b|l|xl)" >&2; exit 1 ;;
esac

EXP_NAME="stage2-llamagen-repa-vq-dinov2-${SIZE}-4m"

echo "[stage2/llamagen-repa] size=${SIZE}(${AR_MODEL},depth${ENCODER_DEPTH}) vq=${VQ_CKPT}"

EVAL_ARGS=()
if [[ "${EVAL_STEPS}" -gt 0 ]]; then
    EVAL_ARGS+=(--eval-steps="${EVAL_STEPS}" --fid-reference-file="${FID_REF}" --eval-keep-samples)
fi

torchrun \
    --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  src/train_ar.py \
    --exp-name="${EXP_NAME}" \
    --data-dir="${DATA_DIR}" \
    --vq-ckpt="${VQ_CKPT}" \
    --ar-model="${AR_MODEL}" \
    --encoder-depth="${ENCODER_DEPTH}" \
    --max-train-steps=4000000 \
    "${EVAL_ARGS[@]}"
