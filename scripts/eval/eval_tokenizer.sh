#!/bin/bash
# ============================================================================
# Evaluate a discrete image tokenizer (reconstruction L1 / PSNR / SSIM / FID
# on the ImageNet-1K val split). Wraps tools/eval_tokenizer.py, which reuses
# the SAME eval helper Stage 0/1 run inline, so the numbers are comparable.
#
# Two axes:
#   TOKENIZER : vq | lfq | ibq   -> tokenizer family + codebook embed dim
#   SOURCE    : official | ours  -> which checkpoint to score
#                 official = the public pretrain you downloaded
#                            (LlamaGen VQ / Open-MAGVIT2 LFQ / TencentARC IBQ)
#                 ours     = a checkpoint from scripts/train/train_tokenizer.sh
#                            or train_gear.sh (carries `vq` + `vq_ema`)
# The checkpoint format is auto-detected by tools/eval_tokenizer.py.
#
# Single node, NPROC_PER_NODE GPUs (set NPROC_PER_NODE=1 for single-GPU).
# ============================================================================

set -e
cd "$(dirname "$0")/../.."   # repo root, so src/... paths resolve

# ---- choose what to evaluate (defaults: vq, ours) --------------------------
TOKENIZER="${TOKENIZER:-vq}"   # vq | lfq | ibq
SOURCE="${SOURCE:-ours}"       # official | ours

# ---- checkpoint paths (edit these) -----------------------------------------
# Official public pretrains (downloaded):
VQ_OFFICIAL_CKPT="/path/to/llamagen/vq_ds16_c2i.pt"              # LlamaGen VQ-16
LFQ_OFFICIAL_CKPT="/path/to/Open-MAGVIT2/pretrain256_16384.ckpt" # TencentARC MAGVIT2 (LFQ)
IBQ_OFFICIAL_CKPT="/path/to/IBQ/IBQ_pretrain_16384.ckpt"         # TencentARC IBQ
# Our own Stage-0 / Stage-1 checkpoint (must match the TOKENIZER family above):
OURS_CKPT="/path/to/checkpoints/0400000.pt"
# For our ckpts, score the EMA shadow (`vq_ema`, usually a bit better) by
# default; set USE_EMA=0 to score the live `vq`. No effect on official ckpts
# (they carry no `vq_ema`, so it silently falls back to their weights).
USE_EMA="${USE_EMA:-1}"

# ---- data / run ------------------------------------------------------------
DATA_DIR="/path/to/imagenet/val"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"   # 0 = full 50k val; e.g. 5000 for a quick smoke test
# Short-side resize interp. 'bicubic' matches our training pipeline; use
# 'bilinear' to reproduce Open-MAGVIT2 / SEED-Voken published numbers.
VAL_RESIZE_MODE="${VAL_RESIZE_MODE:-bicubic}"
OUT_DIR="${OUT_DIR:-}"            # set to dump eval_metrics.json + eval.log

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ===========================================================================
# TOKENIZER -> --vq-model + --codebook-embed-dim
#   VQ=8, LFQ=log2(codebook_size)=14 (forced internally), IBQ=256
# ===========================================================================
case "${TOKENIZER}" in
    vq)  VQ_MODEL="VQ-16";  CODEBOOK_EMBED_DIM=8   ;;
    lfq) VQ_MODEL="LFQ-16"; CODEBOOK_EMBED_DIM=14  ;;
    ibq) VQ_MODEL="IBQ-16"; CODEBOOK_EMBED_DIM=256 ;;
    *) echo "Unknown TOKENIZER='${TOKENIZER}' (vq|lfq|ibq)" >&2; exit 1 ;;
esac

# ===========================================================================
# SOURCE -> checkpoint
# ===========================================================================
case "${SOURCE}" in
    official)
        case "${TOKENIZER}" in
            vq)  CKPT="${VQ_OFFICIAL_CKPT}"  ;;
            lfq) CKPT="${LFQ_OFFICIAL_CKPT}" ;;
            ibq) CKPT="${IBQ_OFFICIAL_CKPT}" ;;
        esac
        ;;
    ours) CKPT="${OURS_CKPT}" ;;
    *) echo "Unknown SOURCE='${SOURCE}' (official|ours)" >&2; exit 1 ;;
esac

echo "[eval-tokenizer] tokenizer=${TOKENIZER}(${VQ_MODEL},embed${CODEBOOK_EMBED_DIM}) source=${SOURCE} use_ema=${USE_EMA}"
echo "[eval-tokenizer] ckpt=${CKPT}"

EXTRA_ARGS=()
[[ "${USE_EMA}" == "1" ]] && EXTRA_ARGS+=(--use-ema)
# Tag the output dir with tokenizer / source / resize-mode / ema so different
# runs never overwrite each other's eval_metrics.json / eval.log.
RUN_TAG="${TOKENIZER}-${SOURCE}-${VAL_RESIZE_MODE}"
if [[ "${USE_EMA}" == "1" ]]; then RUN_TAG="${RUN_TAG}-ema"; else RUN_TAG="${RUN_TAG}-live"; fi
[[ -n "${OUT_DIR}" ]] && EXTRA_ARGS+=(--out-dir="${OUT_DIR}/${RUN_TAG}")

torchrun \
    --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
  tools/eval_tokenizer.py \
    --vq-model="${VQ_MODEL}" \
    --codebook-embed-dim="${CODEBOOK_EMBED_DIM}" \
    --ckpt="${CKPT}" \
    --data-dir="${DATA_DIR}" \
    --image-size="${IMAGE_SIZE}" \
    --val-resize-mode="${VAL_RESIZE_MODE}" \
    --batch-size="${BATCH_SIZE}" \
    --max-samples="${MAX_SAMPLES}" \
    "${EXTRA_ARGS[@]}"
