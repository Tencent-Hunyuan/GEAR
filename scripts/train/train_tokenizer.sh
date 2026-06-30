#!/bin/bash
# ============================================================================
# Stage 0: train a discrete image tokenizer on ImageNet-1K.
#
# One script, three tokenizer families. Pick one with TOKENIZER=vq|lfq|ibq
# (default: vq). The three runs are IDENTICAL except for the quantizer family,
# its codebook embedding dim, and the warm-start checkpoint -- so the only
# thing that changes below is the small per-tokenizer block.
#
# Every other training default (400k steps, lr, batch size, bf16,
# torch.compile, loss config, ...) lives in src/train_tokenizer.py and is
# NOT repeated here -- edit it there if you need to change it.
#
# Single node, 8 GPUs. For multi-node, set NNODES / NODE_RANK / MASTER_ADDR /
# MASTER_PORT (standard torchrun rendezvous) and launch once per node.
# ============================================================================

set -e
cd "$(dirname "$0")/../.."   # repo root, so src/... paths resolve

# ---- choose tokenizer ------------------------------------------------------
TOKENIZER="${TOKENIZER:-vq}"          # vq | lfq | ibq

# ---- paths (edit these) ----------------------------------------------------
# ImageNet-1K train folder, canonical <root>/<synset>/<file>.JPEG layout.
DATA_DIR="/path/to/imagenet/train"
# ImageNet-1K val folder, used for reconstruction eval (L1/PSNR/SSIM/FID).
# Only needed when VQ_EVAL_STEPS > 0.
VAL_DATA_DIR="/path/to/imagenet/val"

# Public pretrained tokenizers to warm-start from. Set to "" to train from
# scratch (the GAN then needs ~5-10k steps to stabilise).
VQ_INIT="/path/to/llamagen/vq_ds16_c2i.pt"               # LlamaGen VQ-16
LFQ_INIT="/path/to/Open-MAGVIT2/pretrain256_16384.ckpt"  # TencentARC MAGVIT2 (LFQ)
IBQ_INIT="/path/to/IBQ/IBQ_pretrain_16384.ckpt"          # TencentARC IBQ

# ---- launch / eval ---------------------------------------------------------
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
VQ_EVAL_STEPS="${VQ_EVAL_STEPS:-25000}"   # reconstruction eval interval; 0 disables

# ===========================================================================
# Per-tokenizer config -- the ONLY thing that differs between vq / lfq / ibq:
#   * vq-model           : family registered in models.Tokenizers
#   * codebook-embed-dim : VQ=8, LFQ=log2(codebook_size)=14, IBQ=256
#   * vq-pretrained-ckpt : the matching public warm-start
# ===========================================================================
case "${TOKENIZER}" in
    vq)
        VQ_MODEL="VQ-16";  CODEBOOK_EMBED_DIM=8;   VQ_PRETRAINED="${VQ_INIT}"
        EXP_NAME="vq16-stage0-bs256-400k" ;;
    lfq)
        VQ_MODEL="LFQ-16"; CODEBOOK_EMBED_DIM=14;  VQ_PRETRAINED="${LFQ_INIT}"
        EXP_NAME="lfq16-stage0-bs256-400k" ;;
    ibq)
        VQ_MODEL="IBQ-16"; CODEBOOK_EMBED_DIM=256; VQ_PRETRAINED="${IBQ_INIT}"
        EXP_NAME="ibq16-stage0-bs256-400k" ;;
    *)
        echo "Unknown TOKENIZER='${TOKENIZER}' (expected vq|lfq|ibq)" >&2; exit 1 ;;
esac

echo "[stage0] tokenizer=${TOKENIZER}  model=${VQ_MODEL}  codebook_embed_dim=${CODEBOOK_EMBED_DIM}"
echo "[stage0] warm-start=${VQ_PRETRAINED:-<scratch>}"

# Reconstruction eval flags are only added when enabled (needs the val folder).
EVAL_ARGS=()
if [[ "${VQ_EVAL_STEPS}" -gt 0 ]]; then
    EVAL_ARGS=(--vq-eval-steps="${VQ_EVAL_STEPS}" --vq-eval-data-dir="${VAL_DATA_DIR}")
fi

torchrun \
    --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  src/train_tokenizer.py \
    --exp-name="${EXP_NAME}" \
    --data-dir="${DATA_DIR}" \
    --vq-model="${VQ_MODEL}" \
    --codebook-embed-dim="${CODEBOOK_EMBED_DIM}" \
    --vq-pretrained-ckpt="${VQ_PRETRAINED}" \
    "${EVAL_ARGS[@]}"
