"""Streaming GPIC (Giant Permissive Image Corpus) t2i dataset.

GPIC (``stanford-vision-lab/gpic``) ships as WebDataset ``.tar`` shards where
each sample is ``{key}.json`` + ``{key}.jpg`` OR ``{key}.png``:

* the image is stored as ``jpg`` or ``png`` (extension varies per sample);
* the caption lives INSIDE the ``json`` field under ``"caption"`` (with a
  ``"caption_type"`` in {tag, short, medium, long}).

Because images mix ``.jpg`` / ``.png`` across shards, HF's webdataset loader
fails (schema CastError) -- so we reuse the direct-tarfile streaming pipeline in
``src/blip3o_dataset.py`` (``_prepare_t2i_dataloader``) and only swap in a
GPIC-specific row extractor that reads raw bytes. All caption types are kept.
The yielded sample contract matches the BLIP3o path.
"""

from __future__ import annotations

import json as _json
from typing import List, Optional, Tuple

from src.blip3o_dataset import _prepare_t2i_dataloader


def gpic_row_extract(sample) -> Tuple[Optional[bytes], str]:
    """GPIC extractor: image bytes (``jpg`` or ``png``) + caption from ``json``.

    Returns ``(image_bytes | None, caption_str)``. Image decoding to PIL is done
    by the stream dataset.
    """
    img = sample.get("jpg")
    if img is None:
        img = sample.get("jpeg") or sample.get("png")
    if img is None:
        return None, ""

    meta = sample.get("json")
    if isinstance(meta, (bytes, bytearray)):
        try:
            meta = _json.loads(meta.decode("utf-8", errors="ignore"))
        except Exception:
            return None, ""
    if not isinstance(meta, dict):
        return None, ""

    caption = meta.get("caption", "")
    if caption is None:
        caption = ""
    if isinstance(caption, (bytes, bytearray)):
        caption = caption.decode("utf-8", errors="ignore")
    return img, str(caption)


def prepare_gpic_t2i_dataloader(
    *,
    data_dir: str,
    tokenizer,
    batch_size: int,
    image_size: int,
    text_max_len: int,
    num_workers: int,
    rank: int,
    world_size: int,
    seed: int = 0,
    random_hflip: bool = True,
    shuffle_buffer_size: Optional[int] = None,
    prefetch_factor: int = 2,
    data_probs: Optional[List[float]] = None,
):
    """Build a streaming GPIC t2i dataloader. Returns ``(dataloader, None)``.

    All caption types (tag/short/medium/long) are kept.
    """
    return _prepare_t2i_dataloader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=batch_size,
        image_size=image_size,
        text_max_len=text_max_len,
        num_workers=num_workers,
        rank=rank,
        world_size=world_size,
        row_extract_fn=gpic_row_extract,
        log_tag="GPIC",
        seed=seed,
        random_hflip=random_hflip,
        shuffle_buffer_size=shuffle_buffer_size,
        prefetch_factor=prefetch_factor,
        data_probs=data_probs,
    )
