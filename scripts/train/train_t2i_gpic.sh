#!/bin/bash
# ============================================================================
# Stage 2 -- text-to-image (t2i) training on GPIC.
#
# Trains the LlamaGen-1B t2i model (dual-stream joint self-attention, frozen
# Qwen text encoder, frozen VQ tokenizer, REPA on) FROM SCRATCH -- there is no
# AR warm start for the GPIC recipe.
#
# gear vs llamagen-repa differ ONLY in which frozen tokenizer is used:
#   * llamagen-repa : a plain Stage-0 tokenizer (scripts/train_tokenizer.sh).
#   * gear (e2e)    : the Stage-1 e2e-tuned tokenizer (scripts/train_gear.sh).
# Everything else is identical, so RECIPE just selects the VQ checkpoint.
#
# Single node, 8 GPUs. For multi-node set NNODES / NODE_RANK / MASTER_ADDR /
# MASTER_PORT and launch once per node.
# ============================================================================

set -e
cd "$(dirname "$0")/../.."   # repo root, so src/... paths resolve

# ---- recipe (default: gear) ------------------------------------------------
RECIPE="${RECIPE:-gear}"   # gear | llamagen-repa  (only changes the frozen VQ ckpt)

# ---- paths (edit these) ----------------------------------------------------
# GPIC WebDataset source (single dir of .tar shards; {key}.json + {key}.jpg/.png).
DATA_DIR="/path/to/gpic/train"
# Frozen tokenizer per recipe (EMA-first; --vq-use-ema is on by default).
VQ_STAGE0_CKPT="/path/to/exps/vq16-stage0-bs256-400k/checkpoints/0400000.pt"            # llamagen-repa
VQ_STAGE1_CKPT="/path/to/exps/stage1-vq-dinov2-xl-temp0.1-coeff0.5-256px/checkpoints/0400000.pt"  # gear

# ---- step / epoch budget ---------------------------------------------------
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-400000}"
EPOCHS="${EPOCHS:-1}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-100000000}"   # GPIC train ~= 100M images

# ---- optional COCO FID/CLIP eval (set COCO_FID_STEPS=0 to skip) -------------
COCO_FID_PATH="/path/to/COCO2017-captions/val"   # dir of parquet (caption,image)
COCO_FID_STEPS="${COCO_FID_STEPS:-50000}"
COCO_FID_NUM="${COCO_FID_NUM:-1000}"
COCO_FID_CFG="${COCO_FID_CFG:-1.75}"

# ---- launch ----------------------------------------------------------------
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ===========================================================================
# RECIPE -> frozen VQ checkpoint
# ===========================================================================
case "${RECIPE}" in
    gear)          VQ_CKPT="${VQ_STAGE1_CKPT}" ;;
    llamagen-repa) VQ_CKPT="${VQ_STAGE0_CKPT}" ;;
    *) echo "Unknown RECIPE='${RECIPE}' (gear|llamagen-repa)" >&2; exit 1 ;;
esac

EXP_NAME="t2i-gpic-${RECIPE}-1b"

echo "[t2i/gpic] recipe=${RECIPE} vq=${VQ_CKPT} steps=${MAX_TRAIN_STEPS}"

EVAL_ARGS=()
if [[ "${COCO_FID_STEPS}" -gt 0 ]]; then
    EVAL_ARGS+=(--coco-fid-steps="${COCO_FID_STEPS}" \
                --coco-fid-dataset-path="${COCO_FID_PATH}" \
                --coco-fid-num-samples="${COCO_FID_NUM}" \
                --coco-fid-cfg-scale="${COCO_FID_CFG}")
fi

# Text encoder (Qwen) and CLIP scorer default to HF-hub ids in
# train_ar_t2i.py; override with --text-encoder / --clip-model-path (or env)
# to point at a local copy.
torchrun \
    --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  src/train_ar_t2i.py \
    --exp-name="${EXP_NAME}" \
    --data-type="gpic" \
    --data-dir="${DATA_DIR}" \
    --vq-ckpt="${VQ_CKPT}" \
    --ar-model="LlamaGen-1B" \
    --encoder-depth=12 \
    --max-train-steps="${MAX_TRAIN_STEPS}" \
    --epochs="${EPOCHS}" \
    --samples-per-epoch="${SAMPLES_PER_EPOCH}" \
    "${EVAL_ARGS[@]}"
