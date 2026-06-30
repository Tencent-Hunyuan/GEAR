"""WISE (WISE_Verified) Stage-1: sample images with the ``src`` t2i AR model.

src replacement for UniWorld's *legacy* WISE ``step1_gen_samples.py``. We
keep the legacy generation layout (one flat image per prompt, named
``<prompt_id>.png``) because that is exactly what the **new** WISE_Verified
evaluator (``step2_vllm_eval.py``) expects: a single directory of
``1.png ... 1000.png``.

Everything downstream is the NEW WISE_Verified pipeline:
    * prompts come from ``data_verified/*_verified.json`` (the refreshed ~200
      prompts + binary protocol), NOT the legacy ``data/*.json``;
    * scoring is done by ``step2_vllm_eval.py`` (Qwen3.5-35B-A3B served by vLLM)
      and ``step3_calculate.py`` (binary WiScore).

Run (single node, 8 GPUs)::

    accelerate launch --num_processes 8 \
        src/eval/wise/step1_gen_samples.py \
        --ckpt-path     /path/to/checkpoints/0400000.pt \
        --vq-ckpt-path  /path/to/vq/0400000.pt \
        --data-dir      src/eval/wise/data_verified \
        --output-dir    /path/to/wise_out/run \
        --cfg-scale 1.75 --resize 512

Model loading / sampling is shared with the GPIC eval via ``src.eval._common``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from glob import glob
from pathlib import Path

from tqdm import tqdm

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))  # src/eval on path -> "_common"

from _common import (  # noqa: E402
    add_model_args,
    generate_images,
    load_model_bundle,
)


def load_wise_prompts(data_dir: Path) -> list[tuple[int, str]]:
    """Read the verified prompt set -> list of (prompt_id, Prompt).

    Globs ``*_verified.json`` so the optional ``merge.json`` is not double
    counted. Each entry has at least ``{Prompt, prompt_id}``.
    """
    files = sorted(glob(str(data_dir / "*_verified.json")))
    if not files:
        raise FileNotFoundError(f"no *_verified.json prompt files in {data_dir}")
    seen: dict[int, str] = {}
    for fp in files:
        with open(fp) as f:
            for item in json.load(f):
                seen[int(item["prompt_id"])] = item["Prompt"]
    return sorted(seen.items())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_model_args(p)
    p.add_argument("--data-dir", type=Path,
                   default=_HERE.parent / "data_verified",
                   help="Dir with WISE_Verified *_verified.json prompt files.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Flat dir where <prompt_id>.png images are written.")
    p.add_argument("--per-proc-batch-size", type=int, default=32)
    p.add_argument("--resize", type=int, default=512,
                   help="Resize each saved PNG to RESIZE x RESIZE.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(args)
    accelerator = bundle.accelerator

    prompts = load_wise_prompts(args.data_dir)

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    # One image per prompt_id; shard round-robin by rank, skip existing.
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    my_prompts = prompts[rank::world_size]
    tasks = [(pid, txt) for (pid, txt) in my_prompts
             if not (args.output_dir / f"{pid}.png").exists()]

    if accelerator.is_main_process:
        print(f"[wise] prompts={len(prompts)} | rank0 todo={len(tasks)}", flush=True)

    n = int(args.per_proc_batch_size)
    n_batches = math.ceil(len(tasks) / n) if tasks else 0
    t0 = time.time()
    for b in tqdm(range(n_batches), disable=not accelerator.is_main_process,
                  desc="wise", unit="batch"):
        batch = tasks[b * n:(b + 1) * n]
        captions = [txt for _, txt in batch]
        images = generate_images(bundle, captions, args, resize=args.resize)
        for (pid, _), img in zip(batch, images):
            img.save(str(args.output_dir / f"{pid}.png"))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"[wise] sampling done in {time.time() - t0:.1f}s -> {args.output_dir}",
              flush=True)


if __name__ == "__main__":
    main()
