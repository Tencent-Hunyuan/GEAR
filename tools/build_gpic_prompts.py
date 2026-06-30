"""Build the GPIC test prompt JSONL for the t2i FD-DINOv2 evaluation.

The GPIC eval reference (``reference_stats/test_stats.npz``) was computed from a
*fixed* 50k subset of the test split, pinned by the ``key`` (50000,) and
``tar_idx`` (50000,) arrays inside that file. To get a **matched** FD-DINOv2
number we must generate exactly one image per key, from that key's own caption.

There is no prebuilt prompt file shipped with the dataset -- the captions live
inside the 128 test tars (``test/gpic_test_{00000..00127}.tar``), one
``{key}.json`` member per image carrying ``caption`` + ``caption_type``. This
script reads the (key, tar_idx) pairs from ``test_stats.npz``, streams only the
matching ``.json`` members out of the relevant tars (image bytes are skipped),
and writes a JSONL of ``{key, caption, caption_type}`` in the SAME ORDER as the
``key`` array in the stats file.

Output line format (mirrors the PixelGen baseline's GPICJsonlDataset)::

    {"key": "...", "caption": "...", "caption_type": "short|medium|long|tag"}

Usage::

    python tools/build_gpic_prompts.py \\
        --stats-npz data/gpic/reference_stats/test_stats.npz \\
        --test-dir  data/gpic/test \\
        --out       data/gpic/gpic_eval_50k.jsonl \\
        --num-workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm


def _tar_path(test_dir: str, tar_idx: int) -> str:
    return os.path.join(test_dir, f"gpic_test_{tar_idx:05d}.tar")


def _extract_one_tar(args) -> tuple[int, dict[str, dict]]:
    """Stream a single test tar and pull captions for the wanted keys.

    Returns ``(tar_idx, {key: {"caption": ..., "caption_type": ...}})``. Only
    ``.json`` members are parsed; image members are skipped without reading
    their payload. Iteration stops early once every wanted key for this tar has
    been found.
    """
    test_dir, tar_idx, wanted_keys = args
    wanted = set(wanted_keys)
    found: dict[str, dict] = {}
    path = _tar_path(test_dir, tar_idx)
    with tarfile.open(path, "r") as tar:
        for member in tar:
            if not member.name.endswith(".json"):
                continue
            key = member.name[: -len(".json")]
            if key not in wanted:
                continue
            meta = json.loads(tar.extractfile(member).read())
            found[key] = {
                "caption": meta.get("caption", ""),
                "caption_type": meta.get("caption_type", ""),
            }
            if len(found) == len(wanted):
                break
    return tar_idx, found


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--stats-npz",
        default="data/gpic/reference_stats/test_stats.npz",
        help="Reference stats npz holding the `key` + `tar_idx` arrays.",
    )
    p.add_argument(
        "--test-dir",
        default="datasets/stanford-vision-lab/gpic/test",
        help="Directory of gpic_test_{00000..00127}.tar shards.",
    )
    p.add_argument(
        "--out",
        default="data/gpic/gpic_eval_50k.jsonl",
        help="Output JSONL path (one {key, caption, caption_type} per line). "
        "Named to match the GPIC PixelGen baseline's gpic_eval_50k.jsonl.",
    )
    p.add_argument(
        "--num-workers", type=int, default=16,
        help="Parallel tar readers (one tar per task). 0/1 = sequential.",
    )
    args = p.parse_args()

    stats = np.load(args.stats_npz)
    if "key" not in stats or "tar_idx" not in stats:
        raise KeyError(
            f"{args.stats_npz} lacks `key`/`tar_idx`; cannot map keys to tars. "
            f"Found: {sorted(stats.keys())}"
        )
    keys = [str(k) for k in stats["key"].tolist()]
    tar_idx = np.asarray(stats["tar_idx"]).astype(int).tolist()
    print(f"loaded {len(keys)} keys from {args.stats_npz}")

    # Group wanted keys per tar so each tar is opened exactly once.
    by_tar: dict[int, list[str]] = {}
    for k, ti in zip(keys, tar_idx):
        by_tar.setdefault(ti, []).append(k)
    print(f"spanning {len(by_tar)} tars (idx {min(by_tar)}..{max(by_tar)})")

    tasks = [(args.test_dir, ti, ks) for ti, ks in sorted(by_tar.items())]
    records: dict[str, dict] = {}

    if args.num_workers and args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(_extract_one_tar, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="tars"):
                _, found = fut.result()
                records.update(found)
    else:
        for t in tqdm(tasks, desc="tars"):
            _, found = _extract_one_tar(t)
            records.update(found)

    missing = [k for k in keys if k not in records]
    if missing:
        print(f"WARNING: {len(missing)} keys had no caption (e.g. {missing[:3]})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_written = 0
    with open(args.out, "w") as f:
        # Preserve the stats `key` ordering for reproducibility.
        for k in keys:
            rec = records.get(k)
            if rec is None:
                continue
            f.write(json.dumps({
                "key": k,
                "caption": rec["caption"],
                "caption_type": rec["caption_type"],
            }) + "\n")
            n_written += 1

    print(f"wrote {n_written} prompts -> {args.out}")


if __name__ == "__main__":
    main()
