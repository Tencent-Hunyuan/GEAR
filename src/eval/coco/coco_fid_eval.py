"""Stand-alone COCO caption->image FID + CLIPScore for src t2i checkpoints.

This is the OFFLINE version of the COCO FID/CLIPScore that
``train_ar_t2i.py`` logs online during training. It reuses the exact same
machinery so the numbers are comparable:

  * model loading + sampling  -> ``src.eval._common`` (shared with the
    GenEval / DPG-Bench / GPIC evals; reads the AR/VQ config from the ckpt args,
    or from CLI overrides for published weights);
  * FID + CLIPScore           -> ``src.coco_fid.CocoFIDEvaluator`` (the
    same class the trainer uses).

Unlike GenEval/DPG-Bench (sample-then-score, two stages/envs), COCO FID is a
single stage: the evaluator drives sampling through a ``caption -> PIL`` callback
and computes FID (vs COCO ground-truth) + CLIPScore in one pass, all in the
``gear`` env -- no extra scorer environment.

Run (8 GPUs, distributed)::

    accelerate launch --num_processes 8 \\
        src/eval/coco/coco_fid_eval.py \\
        --ckpt-path     /path/to/checkpoints/0390625.pt \\
        --vq-ckpt-path  /path/to/vq.pt \\
        --coco-dataset-path /path/to/COCO2017-Val \\
        --cfg-scale 1.5 \\
        --out-dir       /path/to/coco_out/run

The COCO val reference (parquet with ``caption`` + ``image`` columns) is the
same one used for training-time monitoring; download
``BinLin203/COCO2017-Val`` from the Hub.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.coco_fid import CocoFIDEvaluator  # noqa: E402
from src.eval._common import (  # noqa: E402
    add_model_args,
    generate_images,
    load_model_bundle,
    resolve_run_dir,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # ckpt / text-encoder / sampling (cfg-scale, temperature, ...) + arch overrides
    add_model_args(p)

    g = p.add_argument_group("coco fid")
    g.add_argument("--coco-dataset-path", type=str, required=True,
                   help="Dir of COCO val parquet files (columns: caption, image). "
                        "Download BinLin203/COCO2017-Val from the Hub.")
    g.add_argument("--num-samples", type=int, default=30000,
                   help="Number of COCO captions to sample/score (default 30000).")
    g.add_argument("--coco-image-size", type=int, default=256,
                   help="Square size for GT + generated PILs before InceptionV3 "
                        "(FID is conventionally computed at 256).")
    g.add_argument("--coco-seed", type=int, default=42,
                   help="RNG seed for the COCO subset selection.")
    g.add_argument("--inception-batch-size", type=int, default=32)
    g.add_argument("--clip-model-path", type=str, default="openai/clip-vit-base-patch32",
                   help="CLIP model for CLIPScore (local dir or HF id). Empty disables it.")
    g.add_argument("--clip-batch-size", type=int, default=64)
    g.add_argument("--out-dir", type=str, required=True,
                   help="Parent dir; results go to <out-dir>/<exp>/<step>/<cfg-tag>/.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(args)  # AR (EMA) + VQ + frozen Qwen, on this rank
    acc = bundle.accelerator

    evaluator = CocoFIDEvaluator(
        parquet_dir=args.coco_dataset_path,
        num_samples=args.num_samples,
        image_size=args.coco_image_size,
        seed=args.coco_seed,
        inception_batch_size=args.inception_batch_size,
        clip_model_path=(args.clip_model_path or None),
        clip_batch_size=args.clip_batch_size,
    )
    evaluator.setup(bundle.device, torch.float32)

    # caption -> PIL, using the same sampler as every other eval (honours
    # --cfg-scale / --temperature / --top-k / --top-p from add_model_args).
    def _gen(caption: str):
        return generate_images(bundle, [caption], args)[0]

    fid, info = evaluator.evaluate(
        generate_pil_fn=_gen,
        rank=acc.process_index,
        world_size=acc.num_processes,
        verbose=True,
    )

    if acc.is_main_process:
        run_dir = resolve_run_dir(bundle, args, Path(args.out_dir))
        run_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "fid": fid,
            "clip_score": info.get("clip_score"),
            "cfg_scale": args.cfg_scale,
            "num_samples": info.get("n_samples_total"),
            "coco_image_size": args.coco_image_size,
            "ckpt": str(args.ckpt_path),
        }
        with open(run_dir / "coco_fid.json", "w") as f:
            json.dump(result, f, indent=2)
        lines = [f"COCO-FID: {fid}"]
        if info.get("clip_score") is not None:
            lines.append(f"CLIPScore: {info['clip_score']}")
        txt = "\n".join(lines) + "\n"
        (run_dir / "coco_fid.txt").write_text(txt)
        print(f"[coco-eval] wrote {run_dir / 'coco_fid.json'}")
        print(txt, end="")

    acc.wait_for_everyone()


if __name__ == "__main__":
    main()
