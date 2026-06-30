#!/bin/bash
# ============================================================================
# Generation FID (gFID) of a class-conditional AR checkpoint.
#
# Two steps:
#   1. src/inference.py samples FID_NUM images and packs them into a
#      samples.npz under <OUT_DIR>/<EXP_NAME>/step<step>/cfg<g>-temp<g>/.
#   2. tools/evaluator.py (ADM, in the `adm` conda env) computes gFID/sFID/IS/
#      Precision/Recall against the ImageNet reference stats.
#
# VQ weights:
#   * Stage-1 gear (e2e) ckpt already bundles the tokenizer -> leave VQ_CKPT
#     empty (inference.py reads ckpt['vq_ema']/['vq']).
#   * Stage-2 ckpt has no tokenizer -> set VQ_CKPT to the stage-0/stage-1 ckpt.
# ============================================================================

set -e
cd "$(dirname "$0")/../.."   # repo root, so src/... paths resolve

# ---- what to evaluate (edit these, or override inline: CKPT=... bash ...) ---
EXP_NAME="${EXP_NAME:-my-run}"                                       # run name (= <run> in CKPT) + output sub-folder
CKPT="${CKPT:-/path/to/exps/${EXP_NAME}/checkpoints/4000000.pt}"     # AR checkpoint to sample from
VQ_CKPT="${VQ_CKPT:-}"                                               # empty for stage-1 gear; set for stage-2 / HF weights
OUT_DIR="${OUT_DIR:-/path/to/infer_out}"                            # samples written here
REF_NPZ="${REF_NPZ:-/path/to/VIRTUAL_imagenet256_labeled.npz}"       # ADM reference stats

# ---- model architecture ----------------------------------------------------
# REQUIRED for published HuggingFace weights (their args snapshot is stripped on
# upload). For a local training ckpt these can be left as-is -- the ckpt's saved
# args fill them in. IMAGE_SIZE is the resolution the AR was trained at.
# NOTE: all released AR generators are VQ-16, so the tokenizer architecture falls
# back to the canonical VQ-16 defaults -- no need to pass any vq-* flags here.
# Only when evaluating an LFQ/IBQ-based AR would you add (to the inference call):
#   --vq-model LFQ-16 --codebook-embed-dim 14   (or  IBQ-16 / 256).
AR_MODEL="${AR_MODEL:-LlamaGen-XL}"   # LlamaGen-B | LlamaGen-L | LlamaGen-XL
IMAGE_SIZE="${IMAGE_SIZE:-256}"       # 256 | 384 | 512

# ---- sampling knobs --------------------------------------------------------
CFG_SCALE="${CFG_SCALE:-1.0}"
TEMP="${TEMP:-1.0}"
FID_NUM="${FID_NUM:-50000}"
PER_PROC_BS="${PER_PROC_BS:-32}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

# ---- 1. sample + pack samples.npz ------------------------------------------
VQ_ARG=()
[[ -n "${VQ_CKPT}" ]] && VQ_ARG=(--vq-ckpt-path "${VQ_CKPT}")

accelerate launch --num_processes "${NUM_PROCESSES}" --num_machines 1 \
    --main_process_ip 127.0.0.1 --main_process_port 12345 \
  src/inference.py \
    --ckpt-path "${CKPT}" \
    "${VQ_ARG[@]}" \
    --ar-model "${AR_MODEL}" \
    --image-size "${IMAGE_SIZE}" \
    --exp-name "${EXP_NAME}" \
    --output-dir "${OUT_DIR}" \
    --fid-num "${FID_NUM}" \
    --per-proc-batch-size "${PER_PROC_BS}" \
    --cfg-scale "${CFG_SCALE}" \
    --temperature "${TEMP}"

# ---- 2. compute gFID (ADM evaluator, in the `adm` conda env) ----------------
STEP_TAG="step$(basename "${CKPT}" .pt)"
SAMPLING_TAG="$(printf 'cfg%g-temp%g' "${CFG_SCALE}" "${TEMP}")"
SAMPLES_NPZ="${OUT_DIR}/${EXP_NAME}/${STEP_TAG}/${SAMPLING_TAG}/samples.npz"

conda activate adm
python tools/evaluator.py "${REF_NPZ}" "${SAMPLES_NPZ}"
