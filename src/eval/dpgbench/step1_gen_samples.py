"""DPG-Bench Stage-1: sample images with the ``src`` Stage-2 t2i AR model.

src replacement for UniWorld's DPG-Bench ``step1_gen_samples.py``. The
output layout matches what ``step2_compute_dpg_bench.py`` expects: per prompt we
sample ``pic_num`` (default 4) images, tile them into a 2x2 grid, and save it as
``<output_dir>/<key>.png`` (e.g. ``226.png``). step-2 crops each tile back out
using ``--resolution`` (which must equal ``--resize`` here).

Run (single node, 8 GPUs)::

    accelerate launch --num_processes 8 \
        src/eval/dpgbench/step1_gen_samples.py \
        --ckpt-path     /path/to/checkpoints/0400000.pt \
        --vq-ckpt-path  /path/to/vq/0400000.pt \
        --prompts-json  src/eval/dpgbench/eval_prompts/dpgbench_prompts.json \
        --output-dir    /path/to/dpgbench_out \
        --pic-num 4 --resize 512 --cfg-scale 1.75

Model loading / sampling is shared with the GPIC eval via ``src.eval._common``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))  # src/eval on path -> "_common"

from _common import (  # noqa: E402
    add_model_args,
    generate_images,
    load_model_bundle,
)


def tile_grid(images: list[Image.Image], tile: int) -> Image.Image:
    """Tile up to 4 images into a 2x2 grid of side ``2*tile`` (row-major)."""
    grid = Image.new("RGB", (tile * 2, tile * 2))
    for idx, img in enumerate(images[:4]):
        if img.size != (tile, tile):
            img = img.resize((tile, tile), Image.BICUBIC)
        row, col = idx // 2, idx % 2
        grid.paste(img, (col * tile, row * tile))
    return grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_model_args(p)
    p.add_argument("--prompts-json", type=Path,
                   default=_HERE.parent / "eval_prompts" / "dpgbench_prompts.json",
                   help="DPG-Bench prompt JSON ({filename: prompt}).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where the <key>.png 2x2 grids are written.")
    p.add_argument("--pic-num", type=int, default=4,
                   help="Images per prompt tiled into the grid (DPG default = 4).")
    p.add_argument("--per-proc-batch-size", type=int, default=32)
    p.add_argument("--resize", type=int, default=512,
                   help="Per-tile side; pass the SAME value to step2 --resolution.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(args)
    accelerator = bundle.accelerator

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    with open(args.prompts_json) as f:
        data = list(json.load(f).items())

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    my_data = data[rank::world_size]
    # One task per (prompt, sample-slot); the grid is assembled once all
    # pic_num tiles for a prompt are sampled.
    my_data = [(fn, txt) for (fn, txt) in my_data
               if not (args.output_dir / fn.replace(".txt", ".png")).exists()]

    if accelerator.is_main_process:
        print(f"[dpgbench] prompts={len(data)} pic_num={args.pic_num} "
              f"| rank0 todo_prompts={len(my_data)}", flush=True)

    # Expand to a flat (key, caption) task list (pic_num tiles per prompt) so we
    # can batch across prompts; regroup by key when saving.
    flat: list[tuple[str, str]] = []
    for fn, txt in my_data:
        key = fn.replace(".txt", "")
        for _ in range(args.pic_num):
            flat.append((key, txt))

    n = int(args.per_proc_batch_size)
    n_batches = math.ceil(len(flat) / n) if flat else 0
    tiles: dict[str, list[Image.Image]] = {}
    t0 = time.time()
    for b in tqdm(range(n_batches), disable=not accelerator.is_main_process,
                  desc="dpgbench", unit="batch"):
        batch = flat[b * n:(b + 1) * n]
        captions = [c for _, c in batch]
        images = generate_images(bundle, captions, args, resize=args.resize)
        for (key, _), img in zip(batch, images):
            tiles.setdefault(key, []).append(img)
            if len(tiles[key]) == args.pic_num:
                grid = tile_grid(tiles.pop(key), args.resize)
                grid.save(str(args.output_dir / f"{key}.png"))

    # Flush any partially-filled groups (shouldn't happen, but be safe).
    for key, imgs in tiles.items():
        tile_grid(imgs, args.resize).save(str(args.output_dir / f"{key}.png"))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"[dpgbench] sampling done in {time.time() - t0:.1f}s -> {args.output_dir}",
              flush=True)


if __name__ == "__main__":
    main()
