"""Streaming WebDataset text-to-image dataset for the src Stage-2 t2i trainer.

Reads WebDataset ``.tar`` shards DIRECTLY with the stdlib ``tarfile`` (streaming
mode) instead of going through HuggingFace ``datasets``. This is deliberate:

* HF's webdataset loader infers an Arrow schema from the first shard and then
  casts every later shard to it. GPIC mixes ``.jpg`` and ``.png`` images across
  shards, so a png-only shard fails to cast to a jpg-inferred schema
  (``CastError: column names don't match``). A direct tar reader has no global
  schema, so jpg/png/json simply differ per sample with no casting.
* It also sidesteps this repo's local ``datasets/`` package shadowing the
  installed HF ``datasets``.

Sharding is done over (DDP rank x DataLoader worker): each (rank, worker) owns a
strided, disjoint subset of the tar shards, so no sample is seen twice within an
epoch and all workers stay busy. Multiple sources are interleaved by weight.

Output contract (per yielded sample), aligned with the ImageNet path consumed by
``src/train_ar_t2i.py``:

* ``image``    : ``torch.uint8`` tensor ``(3, image_size, image_size)`` in
  ``[0, 255]`` -- the *raw* image; downstream codec / REPA transforms handle the
  domain conversions.
* ``input_ids``: ``torch.long`` ``(text_max_len,)`` -- Qwen tokenizer ids,
  right-padded.
* ``attn_mask``: ``torch.long`` ``(text_max_len,)`` -- 1 real / 0 pad.

The (frozen) Qwen text encoder forward is NOT run here -- it belongs on the GPU
in the training loop; the dataloader only tokenizes on CPU workers.
"""

from __future__ import annotations

import io
import random
import tarfile
from glob import glob
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import transforms

# ---- robustness guards against pathological samples ------------------------
# A single corrupt / huge image or a multi-MB caption can make one rank's data
# worker stall for minutes (PIL decoding a giant image, tokenizing a huge
# string). With per-step DDP collectives that stall becomes a global "hang" on
# the other ranks. These guards turn "very slow" into "fast skip/cap":
#   * tolerate truncated JPEG/PNG instead of erroring deep in the C decoder;
#   * raise (-> caught -> skip) instead of slowly decoding decompression bombs;
#   * cap the raw caption length before tokenizing.
ImageFile.LOAD_TRUNCATED_IMAGES = True
# ~50MP cap (≈7000x7000). Above this PIL raises DecompressionBombError, which
# our per-sample try/except catches and skips (instead of a slow decode).
Image.MAX_IMAGE_PIXELS = 50_000_000
MAX_CAPTION_CHARS = 4000


def _resolve_tar_files(path: str) -> List[str]:
    """Expand a directory / glob / explicit file into a sorted .tar file list.

    * ``"/data/blip3o"``        -> ``/data/blip3o/*.tar``
    * ``"/data/blip3o/*.tar"``  -> glob expanded
    * ``"/data/a.tar"``         -> ``["/data/a.tar"]``
    """
    if any(ch in path for ch in "*?["):
        files = sorted(glob(path, recursive=True))
    elif path.endswith(".tar"):
        files = [path]
    else:
        files = sorted(glob(f"{path.rstrip('/')}/*.tar"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .tar shards matched: {path}")
    return files


def _split_key_ext(member_name: str) -> Tuple[str, str]:
    """Split a webdataset member path into (key, ext).

    ``"a/b/0001.jpg" -> ("a/b/0001", "jpg")``; an extensionless name keeps an
    empty ext. WebDataset groups files of one sample by the shared ``key``.
    """
    base = member_name.rsplit("/", 1)[-1]
    if "." in base:
        # key keeps any leading directory; ext is the final suffix only.
        key, ext = member_name.rsplit(".", 1)
        return key, ext.lower()
    return member_name, ""


def _iter_tar_samples(tar_path: str):
    """Stream one tar, grouping consecutive members by key.

    Yields ``dict`` mapping ``ext -> raw bytes`` plus ``__key__`` / ``__url__``.
    Uses streaming mode (``r|*``) so the tar index is never built (fast cold
    start on network filesystems); the current member is read immediately.
    """
    with tarfile.open(tar_path, "r|*") as tar:
        cur_key = None
        sample: Dict[str, object] = {}
        for member in tar:
            if not member.isfile():
                continue
            key, ext = _split_key_ext(member.name)
            if cur_key is None:
                cur_key = key
            if key != cur_key:
                if sample:
                    sample["__key__"] = cur_key
                    sample["__url__"] = tar_path
                    yield sample
                sample = {}
                cur_key = key
            f = tar.extractfile(member)
            sample[ext] = f.read() if f is not None else b""
        if sample:
            sample["__key__"] = cur_key
            sample["__url__"] = tar_path
            yield sample


def _buffer_shuffle(iterable, buf_size: int, rng: random.Random):
    """Reservoir-style streaming shuffle with a fixed-size buffer."""
    if buf_size <= 1:
        yield from iterable
        return
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= buf_size:
            j = rng.randrange(len(buf))
            buf[j], buf[-1] = buf[-1], buf[j]
            yield buf.pop()
    rng.shuffle(buf)
    yield from buf


def blip3o_row_extract(sample) -> Tuple[Optional[bytes], str]:
    """BLIP3o extractor: ``jpg`` image bytes + ``txt`` caption.

    Returns ``(image_bytes | None, caption_str)``. Image decoding to PIL is
    done by the stream dataset.
    """
    img = sample.get("jpg")
    if img is None:
        img = sample.get("jpeg") or sample.get("png")
    caption = sample.get("txt", b"")
    if isinstance(caption, (bytes, bytearray)):
        caption = caption.decode("utf-8", errors="ignore")
    return img, str(caption or "")


class WebDatasetT2IStream(IterableDataset):
    """Direct-tarfile streaming webdataset -> (raw uint8 image, tokenized caption).

    Format-agnostic via ``row_extract_fn(sample) -> (image_bytes | PIL | None,
    caption_str)`` -- BLIP3o (``jpg``+``txt``) and GPIC (``jpg``/``png`` +
    caption inside ``json``) only differ by their extractor.
    """

    def __init__(
        self,
        sources: List[List[str]],
        tokenizer,
        image_size: int = 256,
        text_max_len: int = 256,
        random_hflip: bool = True,
        row_extract_fn: Optional[Callable] = None,
        probs: Optional[List[float]] = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        shuffle_buffer_size: int = 1024,
        log_tag: str = "WebDatasetT2IStream",
    ):
        self.sources = sources                       # list of per-source tar lists
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.row_extract_fn = row_extract_fn or blip3o_row_extract
        self.probs = probs
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.log_tag = log_tag
        self._iter_count = 0

        tlist = [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
        ]
        if random_hflip:
            tlist.append(transforms.RandomHorizontalFlip(p=0.5))
        tlist.append(transforms.PILToTensor())  # uint8 (3, H, W) in [0, 255]
        self.transform = transforms.Compose(tlist)

    def _tokenize(self, caption: str) -> Dict[str, torch.Tensor]:
        # Cap raw length first: the tokenizer tokenizes the WHOLE string before
        # truncating to max_length, so a multi-MB caption would be slow.
        if len(caption) > MAX_CAPTION_CHARS:
            caption = caption[:MAX_CAPTION_CHARS]
        enc = self.tokenizer(
            caption,
            max_length=self.text_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0].long(),
            "attn_mask": enc["attention_mask"][0].long(),
        }

    def _shard_id_total(self) -> Tuple[int, int]:
        """Global shard index over (rank x worker) and the total shard count."""
        worker = get_worker_info()
        wid = worker.id if worker is not None else 0
        nworkers = worker.num_workers if worker is not None else 1
        return self.rank * nworkers + wid, self.world_size * nworkers

    @staticmethod
    def _assign_tars(tars: List[str], gid: int, total: int) -> List[str]:
        """Assign this (rank, worker) shard ``gid`` of ``total`` its tar subset.

        * #tars >= #shards (the normal large-dataset case): strided, disjoint
          split ``tars[gid::total]`` -- each tar is owned by exactly one shard,
          so no sample is seen twice within a cycle (unchanged behaviour).
        * #tars  < #shards (scarce-data fine-tune, e.g. 11 tars vs 64+ ranks):
          striding would leave most shards with 0 tars -> that rank yields
          nothing -> DDP collective deadlock. Instead we guarantee EVERY shard
          gets exactly one tar via ``tars[gid % #tars]``. Tars are then
          intentionally REPEATED across shards; combined with the per-cycle
          reshuffle + per-shard seed (and random hflip), each shard reads its
          tar in a different order, so the scarce data is reused with variety
          rather than duplicated verbatim. This is the desired behaviour when
          the dataset is deliberately small.
        """
        n = len(tars)
        if n == 0:
            return []
        if n >= total:
            return list(tars[gid::total])
        return [tars[gid % n]]

    def _samples_from_tars(self, tar_list: List[str]):
        for tp in tar_list:
            try:
                yield from _iter_tar_samples(tp)
            except Exception as e:  # skip a corrupt / truncated shard
                print(f"[{self.log_tag}] Skipping tar {tp}: {e}", flush=True)
                continue

    def _interleave(self, src_iters, probs, rng):
        """Weighted single-epoch interleave; drop sources as they exhaust."""
        active = [i for i in range(len(src_iters))]
        weights = list(probs) if probs is not None else [1.0] * len(src_iters)
        while active:
            tot = sum(weights[i] for i in active)
            r = rng.random() * tot
            acc = 0.0
            chosen = active[-1]
            for i in active:
                acc += weights[i]
                if r <= acc:
                    chosen = i
                    break
            try:
                yield next(src_iters[chosen])
            except StopIteration:
                active.remove(chosen)

    def __iter__(self):
        gid, total = self._shard_id_total()
        # This (rank, worker)'s tar subset. When there are at least as many tars
        # as shards, this is a disjoint strided split. When tars are scarcer
        # than shards (e.g. an 11-tar fine-tune set on 64+ ranks), every shard
        # is instead guaranteed one (repeated-across-shards) tar -- see
        # ``_assign_tars`` -- so no rank is ever idle (which would deadlock DDP).
        my_sources = [self._assign_tars(tars, gid, total) for tars in self.sources]
        if sum(len(s) for s in my_sources) == 0:
            # Only reachable if a source genuinely has 0 tars. Returning here
            # would desync DDP (this rank yields nothing); warn loudly.
            print(f"[{self.log_tag}] WARNING: shard {gid}/{total} got 0 tars "
                  f"(a source resolved to an empty tar list). This rank/worker will be idle.")
            return

        base_seed = self.seed + 9176 * gid + 7919 * self._iter_count
        self._iter_count += 1

        # INFINITE cycle: when the shard is exhausted we reshuffle and restart
        # in-process (NO DataLoader re-iter, NO StopIteration leaking to the
        # trainer), so streaming never desyncs DDP at an epoch boundary. The
        # trainer stops purely on a step budget.
        cycle = 0
        while True:
            rng = random.Random(base_seed + 104729 * cycle)
            cycle += 1
            src_iters = []
            for tars in my_sources:
                t = list(tars)
                rng.shuffle(t)                 # reshuffle shard order each cycle
                src_iters.append(self._samples_from_tars(t))

            if len(src_iters) == 1:
                sample_stream = src_iters[0]
            else:
                sample_stream = self._interleave(src_iters, self.probs, rng)
            sample_stream = _buffer_shuffle(sample_stream, self.shuffle_buffer_size, rng)

            produced = 0
            for sample in sample_stream:
                try:
                    img, caption = self.row_extract_fn(sample)
                    if img is None:
                        continue
                    if isinstance(img, Image.Image):
                        image = img
                    else:
                        image = Image.open(io.BytesIO(img))
                    image = image.convert("RGB")

                    pixel_uint8 = self.transform(image)
                    tok = self._tokenize(str(caption))
                    produced += 1
                    yield {
                        "image": pixel_uint8,
                        "input_ids": tok["input_ids"],
                        "attn_mask": tok["attn_mask"],
                    }
                except Exception as e:  # robustness on a single bad sample
                    print(f"[{self.log_tag}] Skipping sample: {e}")
                    continue

            if produced == 0:
                # Every assigned shard yielded nothing usable -> avoid a busy
                # infinite empty loop (which would hang). Stop this worker.
                print(f"[{self.log_tag}] WARNING: shard {gid}/{total} produced 0 "
                      f"usable samples in a full cycle; stopping this worker.")
                return


# Backwards-compatible alias.
BLIP3oT2IStreamDataset = WebDatasetT2IStream


def blip3o_t2i_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "attn_mask": torch.stack([b["attn_mask"] for b in batch], dim=0),
    }


def _prepare_t2i_dataloader(
    *,
    data_dir: str,
    tokenizer,
    batch_size: int,
    image_size: int,
    text_max_len: int,
    num_workers: int,
    rank: int,
    world_size: int,
    row_extract_fn: Callable,
    log_tag: str,
    seed: int = 0,
    random_hflip: bool = True,
    shuffle_buffer_size: Optional[int] = None,
    prefetch_factor: int = 2,
    data_probs: Optional[List[float]] = None,
):
    """Shared builder for the direct-tarfile streaming t2i dataloader."""
    data_dirs = [d.strip() for d in data_dir.split(",") if d.strip()]
    sources = []
    for d in data_dirs:
        files = _resolve_tar_files(d)
        if rank == 0:
            print(f"[{log_tag}] Source {d}: {len(files)} tar shards "
                  f"(first={files[0].split('/')[-1]}, last={files[-1].split('/')[-1]})")
        sources.append(files)

    probs = None
    if data_probs is not None and len(data_dirs) > 1:
        tot = float(sum(data_probs))
        probs = [p / tot for p in data_probs]

    if shuffle_buffer_size is None:
        shuffle_buffer_size = batch_size * 64

    stream_dataset = WebDatasetT2IStream(
        sources=sources,
        tokenizer=tokenizer,
        image_size=image_size,
        text_max_len=text_max_len,
        random_hflip=random_hflip,
        row_extract_fn=row_extract_fn,
        probs=probs,
        rank=rank,
        world_size=world_size,
        seed=seed,
        shuffle_buffer_size=shuffle_buffer_size,
        log_tag=log_tag,
    )

    dataloader = DataLoader(
        stream_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
        collate_fn=blip3o_t2i_collate,
    )
    return dataloader, None


def prepare_blip3o_t2i_dataloader(
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
    """Build a streaming BLIP3o ({key}.jpg + {key}.txt) t2i dataloader.

    ``data_dir`` may be a directory / glob / .tar, or a comma-separated list of
    them for weighted interleaved streaming (see ``data_probs``).
    Returns ``(dataloader, None)``.
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
        row_extract_fn=blip3o_row_extract,
        log_tag="BLIP3o",
        seed=seed,
        random_hflip=random_hflip,
        shuffle_buffer_size=shuffle_buffer_size,
        prefetch_factor=prefetch_factor,
        data_probs=data_probs,
    )
