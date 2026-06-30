# WISE_Verified (src)

WISE for `src` t2i checkpoints. This uses the **new
WISE_Verified** protocol, not the legacy GPT-4o one:

- **Prompts / meta:** `data_verified/*_verified.json` (the refreshed ~200
  prompts, 1000 total), copied from the latest WISE repo.
- **Judge model:** **Qwen3.5-35B-A3B** (`Qwen/Qwen3.5-35B-A3B`) served by a vLLM
  OpenAI-compatible endpoint.
- **Scoring:** binary WiScore via the new `step2_vllm_eval.py`
  (= upstream `vllm_eval.py`) and `step3_calculate.py`
  (= upstream `calculate_verified.py`), both copied **unchanged**.

Only `step1` is src-specific. The on-disk layout follows the legacy
UniWorld WISE (one flat `<prompt_id>.png` per prompt) because that is exactly
what the new judge expects: a single dir of `1.png … 1000.png`.

| step | file | env | output |
|------|------|-----|--------|
| 1. generate | `step1_gen_samples.py` | `gear` | `<run>/<prompt_id>.png` |
| 2. judge    | `step2_vllm_eval.py`   | `gear` + `vllm` | `<run>/Results-qwen35/*_scores_results.jsonl` |
| 3. summary  | `step3_calculate.py`   | any python | `<run>/Results-qwen35/wise_summary.txt` |

The judge needs vLLM (any env, the `gear` env is fine):

```bash
pip install vllm
```

## One-shot (recommended)

Edit the paths at the top of [`scripts/eval/eval_wise.sh`](../../../scripts/eval/eval_wise.sh)
(`CKPT`, `VQ_CKPT`, `OUTPUT_DIR`, `QWEN_MODEL_PATH`) then run it. It samples in the
`gear` env, launches a single-GPU vLLM judge in FAST_START mode, scores all three
categories, and aggregates the WiScore. Architecture knobs are env-overridable:

```bash
AR_MODEL=LlamaGen-XL IMAGE_SIZE=512 TEXT_ENCODER=Qwen/Qwen3-1.7B CFG_SCALE=4.0 \
bash scripts/eval/eval_wise.sh
```

For published HF weights set `VQ_CKPT` to the paired tokenizer; `QWEN_MODEL_PATH`
can be the HF id `Qwen/Qwen3.5-35B-A3B` or a local copy.

## Manual, step by step

### Step 1 — generate (gear env)

Published checkpoints have their `args` stripped, so pass `--ar-model` /
`--image-size` / `--text-encoder` explicitly:

```bash
accelerate launch --num_processes 8 \
    src/eval/wise/step1_gen_samples.py \
    --ckpt-path     /path/to/blip3o-512-ft.pt \
    --vq-ckpt-path  /path/to/vq.pt \
    --ar-model      LlamaGen-XL --image-size 512 \
    --text-encoder  Qwen/Qwen3-1.7B \
    --data-dir      src/eval/wise/data_verified \
    --output-dir    /path/to/wise_out/run \
    --cfg-scale 1.75 --resize 512
```

### Serve the judge (Qwen3.5-35B-A3B via vLLM)

**Single GPU, no tensor-parallel, FAST_START mode** (Triton MoE + GDN, autotune off,
enforce-eager) — the judge fits on one GPU and the server is up in ~2-3 min with no
CUTLASS JIT compile:

```bash
pip install vllm
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-35B-A3B \
    --served-model-name Qwen3.5-35B-A3B \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 \
    --moe-backend triton --gdn-prefill-backend triton \
    --no-enable-flashinfer-autotune --enforce-eager
```

The one-shot `scripts/eval/eval_wise.sh` launches exactly this for you and tears it
down at the end.

### Step 2 — judge each category

```bash
export no_proxy=127.0.0.1,localhost   # so localhost calls bypass any HTTP proxy
RUN=/path/to/wise_out/run
RES=${RUN}/Results-qwen35
for cat in cultural_common_sense spatio-temporal_reasoning natural_science; do
  python src/eval/wise/step2_vllm_eval.py \
      --json_path     src/eval/wise/data_verified/${cat}_verified.json \
      --output_dir    ${RES}/${cat} \
      --image_dir     ${RUN} \
      --api_base      http://127.0.0.1:8000/v1 \
      --api_key       EMPTY \
      --model         Qwen3.5-35B-A3B \
      --result_full   ${RES}/${cat}_full_results.json \
      --result_scores ${RES}/${cat}_scores_results.jsonl \
      --max_workers   96
done
```

### Step 3 — summary (binary WiScore)

```bash
python src/eval/wise/step3_calculate.py \
    ${RES}/cultural_common_sense_scores_results.jsonl \
    ${RES}/natural_science_scores_results.jsonl \
    ${RES}/spatio-temporal_reasoning_scores_results.jsonl \
    --category all
```

`Overall = 0.40*CULTURE + 0.12*(TIME+SPACE+BIOLOGY+PHYSICS+CHEMISTRY)`.
