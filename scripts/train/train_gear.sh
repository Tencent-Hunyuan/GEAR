#!/bin/bash
# ============================================================================
# Stage 1: jointly train the VQ tokenizer + LlamaGen AR with REPA alignment.
#
# One script, all the knobs. Everything is selected at the top; the small
# per-axis blocks below translate each choice into the handful of flags that
# actually differ. Every other hyper-parameter (steps, lr, batch size, bf16,
# torch.compile, loss config, projector dim, ...) lives in
# src/train_gear.py and is NOT repeated here.
#
# Axes:
#   ENCODER    : REPA teacher        dinov2 | dinov3 | jepa21 | siglip2 | clip
#   SIZE       : AR model + REPA tap  b | l | xl   (encoder-depth 4 / 8 / 12)
#   TOKENIZER  : discrete tokenizer   vq | lfq | ibq
#   TEMPERATURE: soft-quant temp T    0.1 | 0.05 | 0.01 | 0.005
#   VAE_ALIGN_PROJ_COEFF: VQ-side REPA loss weight   0.25 | 0.5 | 0.75 | 1.0
#   IMAGE_SIZE : resolution           256 | 384 | 512
#
# Spatial norm: dinov2 / dinov3 align with vanilla REPA (no spatial norm);
# siglip2 / jepa21 / clip turn on iREPA-style zscore spatial norm
# (alpha=SPNORM_ALPHA, default 0.6) which those teachers need to be good
# REPA targets.
#
# Single node, 8 GPUs. For multi-node set NNODES / NODE_RANK / MASTER_ADDR /
# MASTER_PORT (standard torchrun rendezvous) and launch once per node.
# ============================================================================

set -e
cd "$(dirname "$0")/../.."   # repo root, so src/... paths resolve

# ---- choose what to train (defaults: dinov2, xl, vq, T=0.1, coeff=0.5, 256) -
ENCODER="${ENCODER:-dinov2}"                          # dinov2|dinov3|jepa21|siglip2|clip
SIZE="${SIZE:-xl}"                                    # b|l|xl
TOKENIZER="${TOKENIZER:-vq}"                          # vq|lfq|ibq
TEMPERATURE="${TEMPERATURE:-0.1}"                     # 0.1|0.05|0.01|0.005
VAE_ALIGN_PROJ_COEFF="${VAE_ALIGN_PROJ_COEFF:-0.5}"  # 0.25|0.5|0.75|1.0
IMAGE_SIZE="${IMAGE_SIZE:-256}"                       # 256|384|512
SPNORM_ALPHA="${SPNORM_ALPHA:-0.6}"                   # spatial-norm mean strength

# ---- paths (edit these) ----------------------------------------------------
# ImageNet-1K train folder, canonical <root>/<synset>/<file>.JPEG layout.
DATA_DIR="/path/to/imagenet/train"
# Per-tokenizer Stage-0 warm start (the checkpoint produced by
# scripts/train_tokenizer.sh for the matching tokenizer).
VQ_STAGE0_CKPT="/path/to/exps/vq16-stage0-bs256-400k/checkpoints/0400000.pt"
LFQ_STAGE0_CKPT="/path/to/exps/lfq16-stage0-bs256-400k/checkpoints/0400000.pt"
IBQ_STAGE0_CKPT="/path/to/exps/ibq16-stage0-bs256-400k/checkpoints/0400000.pt"

# ---- optional eval (set the *_STEPS to 0 to skip; both need their path) -----
FID_REF="/path/to/VIRTUAL_imagenet256_labeled.npz"  # AR-generation FID ref stats (always 256)
VAL_DATA_DIR="/path/to/imagenet/val"                # VQ reconstruction eval set
EVAL_STEPS="${EVAL_STEPS:-50000}"                   # AR FID cadence (0 disables)
VQ_EVAL_STEPS="${VQ_EVAL_STEPS:-50000}"             # VQ recon eval cadence (0 disables)

# ---- launch ----------------------------------------------------------------
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ===========================================================================
# ENCODER -> --enc-type (+ spatial norm for siglip2/jepa21/clip)
# ===========================================================================
case "${ENCODER}" in
    dinov2)  ENC_TYPE="dinov2-vit-b"; SPNORM_MODE="none"   ;;
    dinov3)  ENC_TYPE="dinov3-vit-b"; SPNORM_MODE="none"   ;;
    jepa21)  ENC_TYPE="jepa21-vit-b"; SPNORM_MODE="zscore" ;;
    siglip2) ENC_TYPE="siglip2-vit-b"; SPNORM_MODE="zscore" ;;
    clip)    ENC_TYPE="clip-vit-L";   SPNORM_MODE="zscore" ;;
    *) echo "Unknown ENCODER='${ENCODER}' (dinov2|dinov3|jepa21|siglip2|clip)" >&2; exit 1 ;;
esac

# ===========================================================================
# SIZE -> --ar-model + --encoder-depth (the REPA tap layer)
# ===========================================================================
case "${SIZE}" in
    b)  AR_MODEL="LlamaGen-B";  ENCODER_DEPTH=4  ;;
    l)  AR_MODEL="LlamaGen-L";  ENCODER_DEPTH=8  ;;
    xl) AR_MODEL="LlamaGen-XL"; ENCODER_DEPTH=12 ;;
    *) echo "Unknown SIZE='${SIZE}' (b|l|xl)" >&2; exit 1 ;;
esac

# ===========================================================================
# TOKENIZER -> --vq-model + --codebook-embed-dim + Stage-0 warm-start ckpt
#   VQ=8, LFQ=log2(codebook_size)=14 (forced internally), IBQ=256
# ===========================================================================
case "${TOKENIZER}" in
    vq)  VQ_MODEL="VQ-16";  CODEBOOK_EMBED_DIM=8;   STAGE0_CKPT="${VQ_STAGE0_CKPT}"  ;;
    lfq) VQ_MODEL="LFQ-16"; CODEBOOK_EMBED_DIM=14;  STAGE0_CKPT="${LFQ_STAGE0_CKPT}" ;;
    ibq) VQ_MODEL="IBQ-16"; CODEBOOK_EMBED_DIM=256; STAGE0_CKPT="${IBQ_STAGE0_CKPT}" ;;
    *) echo "Unknown TOKENIZER='${TOKENIZER}' (vq|lfq|ibq)" >&2; exit 1 ;;
esac

EXP_NAME="stage1-${TOKENIZER}-${ENCODER}-${SIZE}-temp${TEMPERATURE}-coeff${VAE_ALIGN_PROJ_COEFF}-${IMAGE_SIZE}px"

echo "[stage1] encoder=${ENCODER}(${ENC_TYPE}) spnorm=${SPNORM_MODE} size=${SIZE}(${AR_MODEL},depth${ENCODER_DEPTH})"
echo "[stage1] tokenizer=${TOKENIZER}(${VQ_MODEL},embed${CODEBOOK_EMBED_DIM}) T=${TEMPERATURE} coeff=${VAE_ALIGN_PROJ_COEFF} res=${IMAGE_SIZE}"
echo "[stage1] warm-start=${STAGE0_CKPT}"

# Spatial norm only for the teachers that need it.
SPNORM_ARGS=()
if [[ "${SPNORM_MODE}" != "none" ]]; then
    SPNORM_ARGS=(--repa-target-spnorm="${SPNORM_MODE}" --repa-spnorm-alpha="${SPNORM_ALPHA}")
fi

# Eval flags appended only when enabled (each needs its dataset path).
EVAL_ARGS=()
if [[ "${EVAL_STEPS}" -gt 0 ]]; then
    EVAL_ARGS+=(--eval-steps="${EVAL_STEPS}" --fid-reference-file="${FID_REF}" --eval-keep-samples)
fi
if [[ "${VQ_EVAL_STEPS}" -gt 0 ]]; then
    EVAL_ARGS+=(--vq-eval-steps="${VQ_EVAL_STEPS}" --vq-eval-data-dir="${VAL_DATA_DIR}")
fi

torchrun \
    --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  src/train_gear.py \
    --exp-name="${EXP_NAME}" \
    --data-dir="${DATA_DIR}" \
    --image-size="${IMAGE_SIZE}" \
    --vq-model="${VQ_MODEL}" \
    --codebook-embed-dim="${CODEBOOK_EMBED_DIM}" \
    --stage0-init-ckpt="${STAGE0_CKPT}" \
    --ar-model="${AR_MODEL}" \
    --encoder-depth="${ENCODER_DEPTH}" \
    --enc-type="${ENC_TYPE}" \
    --temperature="${TEMPERATURE}" \
    --vae-align-proj-coeff="${VAE_ALIGN_PROJ_COEFF}" \
    "${SPNORM_ARGS[@]}" \
    "${EVAL_ARGS[@]}"
