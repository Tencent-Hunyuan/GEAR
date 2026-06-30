# GenEval (src)

GenEval for `src` t2i checkpoints. The metric code (`step2`/`step3`)
is the original [GenEval](https://github.com/djghosh13/geneval) ported via
UniWorld and is **unchanged** — it only consumes generated images. Only `step1`
was rewritten to sample from our AR+VQ model (shared with the GPIC eval through
`src/eval/_common.py`, which imports the loaders from
`src/inference_t2i.py`).

Like the other eval folders this is a 3-step pipeline:

| step | file | env | output |
|------|------|-----|--------|
| 1. generate | `step1_gen_samples.py` | `gear` | `<run>/<idx>/samples/*.png` |
| 2. detect   | `step2_run_geneval.py` | `geneval_eval` | `<run>/geneval.jsonl` |
| 3. summary  | `step3_summary_score.py` | `geneval_eval` | `<run>/geneval.txt` |

## Setup (do this first)

Sampling (step 1) runs in the `gear` env. The scorer (steps 2-3) needs a separate
`geneval_eval` env (mmcv / mmdet / open_clip) plus two model downloads. Build it once:

```bash
conda create -n geneval_eval python=3.10 -y
conda activate geneval_eval

# 1) torch (cu118)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 2) mmcv-full 1.7.2.
#    On Hopper GPUs (sm_90, e.g. H20) the prebuilt wheels lack sm_90 kernels and
#    step 2 fails with "no kernel image is available". Build from source for 9.0:
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install -y -c conda-forge gxx_linux-64=11
MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="9.0" \
    pip install --no-binary mmcv-full mmcv-full==1.7.2
#    (non-Hopper GPUs can use the prebuilt wheel instead:
#     pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html)

# 3) GenEval python deps + mmdetection (2.x branch)
pip install -r src/eval/geneval/requirements.txt
pip install "numpy<2"          # mmcv/mmdet are built against NumPy 1.x
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection && git checkout 2.x && python setup.py develop && cd ..

# 4) Mask2Former detector + CLIP weights (download once)
DETECTOR_PATH=/path/to/detector ; mkdir -p $DETECTOR_PATH
wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth \
  -O $DETECTOR_PATH/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
CACHE_DIR=/path/to/clip ; mkdir -p $CACHE_DIR
wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt \
  -O $CACHE_DIR/ViT-L-14.pt
```

Point the script's `DETECTOR_PATH` / `CACHE_DIR` at these.

## One-shot (recommended)

Edit the paths at the top of [`scripts/eval/eval_geneval.sh`](../../../scripts/eval/eval_geneval.sh)
(`CKPT`, `VQ_CKPT`, `OUTPUT_DIR`, `DETECTOR_PATH`, `CACHE_DIR`) then run it. It samples
in the `gear` env and runs detection/summary in the `geneval_eval` env. Architecture
knobs are env-overridable:

```bash
AR_MODEL=LlamaGen-XL IMAGE_SIZE=512 TEXT_ENCODER=Qwen/Qwen3-1.7B CFG_SCALE=4.0 \
bash scripts/eval/eval_geneval.sh
```

Point `PROMPTS_JSONL` at `evaluation_metadata_long.jsonl` for the LLM-rewritten set.
For published HF weights set `VQ_CKPT` to the paired tokenizer.

## Manual, step by step

### Step 1 — generate (gear env)

Published checkpoints have their `args` stripped, so pass `--ar-model` /
`--image-size` / `--text-encoder` explicitly:

```bash
accelerate launch --num_processes 8 \
    src/eval/geneval/step1_gen_samples.py \
    --ckpt-path     /path/to/blip3o-512-ft.pt \
    --vq-ckpt-path  /path/to/vq.pt \
    --ar-model      LlamaGen-XL --image-size 512 \
    --text-encoder  Qwen/Qwen3-1.7B \
    --prompts-jsonl src/eval/geneval/eval_prompts/evaluation_metadata.jsonl \
    --output-dir    /path/to/geneval_out/run \
    --n-samples 4 --cfg-scale 1.75 --resize 512
```

For LLM-rewritten prompts, swap in `eval_prompts/evaluation_metadata_long.jsonl`.

### Step 2 — detection (geneval_eval env)

Uses the env + weights from [Setup](#setup-do-this-first):

```bash
conda activate geneval_eval
CUDA_VISIBLE_DEVICES=0 CACHE_DIR=$CACHE_DIR python src/eval/geneval/step2_run_geneval.py \
    /path/to/geneval_out/run \
    --outfile /path/to/geneval_out/run/geneval.jsonl \
    --model-path $DETECTOR_PATH
```

### Step 3 — summary

```bash
python src/eval/geneval/step3_summary_score.py \
    /path/to/geneval_out/run/geneval.jsonl > /path/to/geneval_out/run/geneval.txt
cat /path/to/geneval_out/run/geneval.txt
```
