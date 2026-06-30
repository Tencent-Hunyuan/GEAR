"""GenEval Stage-1: sample images with the ``src`` Stage-2 t2i AR model.

This is the src replacement for UniWorld's GenEval ``step1_gen_samples.py``
(which was hard-wired to FLUX + Qwen). The on-disk layout is kept identical so
the downstream ``step2_run_geneval.py`` / ``step3_summary_score.py`` work
unchanged:

    <output_dir>/<index:05d>/metadata.jsonl
    <output_dir>/<index:05d>/samples/<sample:05d>.png

Run (single node, 8 GPUs)::

    accelerate launch --num_processes 8 \
        src/eval/geneval/step1_gen_samples.py \
        --ckpt-path     /path/to/checkpoints/0400000.pt \
        --vq-ckpt-path  /path/to/vq/0400000.pt \
        --prompts-jsonl src/eval/geneval/eval_prompts/evaluation_metadata.jsonl \
        --output-dir    /path/to/geneval_out \
        --n-samples 4 --cfg-scale 1.75

Model loading / sampling is shared with the GPIC eval via ``src.eval._common``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))  # src/eval on path -> "_common"

from _common import (  # noqa: E402
    add_model_args,
    generate_images,
    load_model_bundle,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_model_args(p)
    p.add_argument("--prompts-jsonl", type=Path,
                   default=_HERE.parent / "eval_prompts" / "evaluation_metadata.jsonl",
                   help="GenEval metadata JSONL (one {tag, include, prompt} per line).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where the <index>/samples/<n>.png tree is written.")
    p.add_argument("--n-samples", type=int, default=4,
                   help="Images sampled per prompt (GenEval default = 4).")
    p.add_argument("--per-proc-batch-size", type=int, default=32)
    p.add_argument("--resize", type=int, default=512,
                   help="Resize each saved PNG to RESIZE x RESIZE (GenEval uses 512).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(args)
    accelerator = bundle.accelerator

    with open(args.prompts_jsonl) as f:
        metadatas = [json.loads(line) for line in f if line.strip()]

    # Main process lays out the per-prompt dirs + metadata; ranks then fill them.
    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for index, metadata in enumerate(metadatas):
            outpath = args.output_dir / f"{index:0>5}"
            (outpath / "samples").mkdir(parents=True, exist_ok=True)
            with open(outpath / "metadata.jsonl", "w") as fp:
                json.dump(metadata, fp)
    accelerator.wait_for_everyone()

    # Flat task list: (caption, destination png). Sharded round-robin by rank.
    tasks: list[tuple[str, Path]] = []
    for index, metadata in enumerate(metadatas):
        sample_dir = args.output_dir / f"{index:0>5}" / "samples"
        for s in range(args.n_samples):
            tasks.append((metadata["prompt"], sample_dir / f"{s:05d}.png"))

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    my_tasks = tasks[rank::world_size]
    my_tasks = [(c, p) for (c, p) in my_tasks if not p.exists()]

    if accelerator.is_main_process:
        print(f"[geneval] prompts={len(metadatas)} n_samples={args.n_samples} "
              f"total_images={len(tasks)} | rank0 todo={len(my_tasks)}", flush=True)

    n = int(args.per_proc_batch_size)
    n_batches = math.ceil(len(my_tasks) / n) if my_tasks else 0
    t0 = time.time()
    for b in tqdm(range(n_batches), disable=not accelerator.is_main_process,
                  desc="geneval", unit="batch"):
        batch = my_tasks[b * n:(b + 1) * n]
        captions = [c for c, _ in batch]
        images = generate_images(bundle, captions, args, resize=args.resize)
        for (_, out_png), img in zip(batch, images):
            img.save(str(out_png))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"[geneval] sampling done in {time.time() - t0:.1f}s -> {args.output_dir}",
              flush=True)


if __name__ == "__main__":
    main()
