#!/bin/bash
# ============================================================================
# Stage 2 -- GEAR (our end-to-end recipe).
#
# Trains a LlamaGen AR + REPA on top of the FROZEN e2e-tuned tokenizer that
# came out of Stage 1 (train_gear.py co-trains VQ + AR). The frozen
# tokenizer is auto-extracted from the Stage-1 ckpt (ckpt["vq_ema"], EMA-first
# via --vq-use-ema).
#
# Two modes (GEAR_MODE):
#   (1) ar-init  [default]  Warm-start the AR from the SAME Stage-1 ckpt (the
#                           AR already trained 400k steps in Stage 1), so we
#                           only need 3.6M more here -> 4M total for the AR.
#                           --vq-override-from-ar-init (on by default) makes
#                           the frozen tokenizer come from that same ckpt, so
#                           AR + tokenizer are guaranteed to match.
#   (2) vq-only             Only freeze the Stage-1 tokenizer; train a fresh
#                           AR from scratch for the full 4M steps (no AR warm
#                           start).
#
# Fixed for this script: REPA teacher = dinov2, tokenizer = vq. Only the AR
# model size and the mode above are selectable.
#
# Single node, 8 GPUs. For multi-node set NNODES / NODE_RANK / MASTER_ADDR /
# MASTER_PORT and launch once per node.
# ============================================================================

set -e
cd "$(dirname "$0")/../.."   # repo root, so src/... paths resolve

# ---- choose size + mode (defaults: xl, ar-init) ----------------------------
SIZE="${SIZE:-xl}"            # b | l | xl   (AR model + REPA tap depth 4 / 8 / 12)
GEAR_MODE="${GEAR_MODE:-ar-init}"   # ar-init (default) | vq-only

# ---- paths (edit these) ----------------------------------------------------
DATA_DIR="/path/to/imagenet/train"
# Stage-1 checkpoint for the chosen size (output of scripts/train_gear.sh).
# In ar-init mode this provides BOTH the AR warm start and the frozen
# tokenizer. In vq-only mode only its tokenizer (vq_ema) is used.
STAGE1_CKPT="/path/to/exps/stage1-vq-dinov2-${SIZE}-temp0.1-coeff0.5-256px/checkpoints/0400000.pt"

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

# ===========================================================================
# GEAR_MODE -> AR warm start + step budget
# ===========================================================================
INIT_ARGS=()
case "${GEAR_MODE}" in
    ar-init)
        # Inherit the Stage-1 AR (already 400k steps) -> 3.6M more = 4M total.
        INIT_ARGS+=(--ar-init-ckpt="${STAGE1_CKPT}")
        MAX_TRAIN_STEPS=3600000
        ;;
    vq-only)
        # Fresh AR, frozen Stage-1 tokenizer only -> full 4M steps.
        MAX_TRAIN_STEPS=4000000
        ;;
    *) echo "Unknown GEAR_MODE='${GEAR_MODE}' (ar-init|vq-only)" >&2; exit 1 ;;
esac

EXP_NAME="stage2-gear-${GEAR_MODE}-vq-dinov2-${SIZE}"

echo "[stage2/gear] mode=${GEAR_MODE} size=${SIZE}(${AR_MODEL},depth${ENCODER_DEPTH}) steps=${MAX_TRAIN_STEPS}"
echo "[stage2/gear] stage1-ckpt=${STAGE1_CKPT}"

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
    --vq-ckpt="${STAGE1_CKPT}" \
    --ar-model="${AR_MODEL}" \
    --encoder-depth="${ENCODER_DEPTH}" \
    --max-train-steps="${MAX_TRAIN_STEPS}" \
    "${INIT_ARGS[@]}" \
    "${EVAL_ARGS[@]}"
