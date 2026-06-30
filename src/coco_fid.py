"""COCO-caption FID evaluator for the text-to-image AR trainer.

Ported (almost verbatim) from the unified-AR trainer's
``utils/eval/coco_fid.py``; the only change is the FID-utility import, which
points at this repo's ``tools/calculate_fid`` (it exposes the same
``InceptionV3`` + ``calculate_frechet_distance`` as the original).

End-to-end shape::

    captions[i], gt_image[i]
        |- extract_features(gt_image[i])               -> ref_feats[i]   (cached once per run)
        '- generator(captions[i], cfg_scale)           -> gen_image[i]
                '- extract_features(gen_image[i])      -> gen_feats[i]
    all_gather(ref_feats), all_gather(gen_feats)       -> [N, 2048] each
    FID = frechet(mu_ref, sigma_ref ; mu_gen, sigma_gen)

Per-rank reference features are computed once (they do not depend on model
weights) and cached for the whole run; generated features are recomputed
every eval. All ranks run the same number of ``generate()`` calls (shard
padded to ``ceil(N / world_size)``) so collective ops don't deadlock.

Both reference (COCO) and generated PILs go through the same aspect-preserving
pipeline before InceptionV3: short-side resize to ``image_size`` then
center-crop to ``(image_size, image_size)``.
"""
from __future__ import annotations

import io
import math
import os
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

from tools.calculate_fid import InceptionV3, calculate_frechet_distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _all_gather_np_2d(local: np.ndarray) -> np.ndarray:
    """All-gather a 2D ``[n_local, D]`` numpy array along dim 0.

    Pads each rank's tensor to the max rank-local length so we can use a
    fixed-size ``all_gather``; trailing pads are dropped on assembly.
    """
    if local.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {local.shape}")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size == 1:
        return local.copy()

    n_local = local.shape[0]
    feat_dim = local.shape[1]
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    size_t = torch.tensor([n_local], device=device, dtype=torch.long)
    sizes = [torch.zeros_like(size_t) for _ in range(world_size)]
    dist.all_gather(sizes, size_t)
    sizes = [int(s.item()) for s in sizes]
    max_n = max(sizes)

    pad_n = max_n - n_local
    flat = torch.from_numpy(local).to(device)
    if pad_n > 0:
        flat = torch.cat(
            [flat, torch.zeros(pad_n, feat_dim, dtype=flat.dtype, device=device)], dim=0
        )
    gather = [torch.empty_like(flat) for _ in range(world_size)]
    dist.all_gather(gather, flat)
    chunks = [g[:n].cpu().numpy() for g, n in zip(gather, sizes)]
    return np.concatenate(chunks, axis=0)


def _fmt_dur(seconds: float) -> str:
    """Pretty-print a wall-time duration (``HhMMmSSs`` / ``MMmSSs`` / ``SSs``)."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:5.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m):d}m{int(s):02d}s"
    h, m = divmod(m, 60)
    return f"{int(h):d}h{int(m):02d}m{int(s):02d}s"


def _decode_coco_image(raw) -> Image.Image:
    """Decode the HF-style ``{"bytes": ..., "path": ...}`` image cell."""
    if isinstance(raw, dict):
        b = raw.get("bytes")
        if b is not None:
            return Image.open(io.BytesIO(b)).convert("RGB")
        p = raw.get("path")
        if p:
            return Image.open(p).convert("RGB")
    if isinstance(raw, (bytes, bytearray)):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    raise ValueError(f"unsupported coco image cell type: {type(raw)}")


def _pil_short_resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    """Aspect-preserving FID preprocessing: short-side resize then center crop."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if w == size and h == size:
        return img
    short = min(w, h)
    if short != size:
        scale = size / short
        new_w = max(size, int(round(w * scale)))
        new_h = max(size, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.BICUBIC)
        w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _pils_to_tensor_01(pils: List[Image.Image], device, dtype) -> torch.Tensor:
    """Stack a list of equal-size PILs to ``[N, 3, H, W]`` in ``[0, 1]``."""
    if not pils:
        return torch.zeros(0, 3, 1, 1, device=device, dtype=dtype)
    arrs = [np.asarray(p, dtype=np.uint8) for p in pils]
    t = torch.from_numpy(np.stack(arrs, axis=0)).to(device)
    return t.permute(0, 3, 1, 2).contiguous().to(dtype) / 255.0


# ---------------------------------------------------------------------------
# CocoFIDEvaluator
# ---------------------------------------------------------------------------
class CocoFIDEvaluator:
    """Compute FID between AR-generated images and COCO ground-truth images.

    Stateful across eval calls in one run: the COCO subset + InceptionV3 are
    loaded once on first ``setup()``; per-rank reference features are computed
    once on first ``evaluate()`` and cached.
    """

    def __init__(
        self,
        parquet_dir: str,
        num_samples: int = 1000,
        image_size: int = 256,
        seed: int = 42,
        inception_batch_size: int = 32,
        clip_model_path: Optional[str] = None,
        clip_batch_size: int = 64,
    ):
        self.parquet_dir = parquet_dir
        self.num_samples = int(num_samples)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.inception_batch_size = int(inception_batch_size)
        # Optional CLIPScore (image-text alignment of generated samples).
        self.clip_model_path = clip_model_path
        self.clip_batch_size = int(clip_batch_size)

        self._captions: Optional[List[str]] = None
        self._gt_images: Optional[List[Image.Image]] = None
        self._inception: Optional[torch.nn.Module] = None
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None
        self._ref_feats_local: Optional[np.ndarray] = None
        self._clip_model = None
        self._clip_processor = None

    # ------------------------------------------------------------------
    # Setup (lazy)
    # ------------------------------------------------------------------
    def _load_coco_subset(self):
        import pyarrow.parquet as pq

        files = sorted(
            os.path.join(self.parquet_dir, f)
            for f in os.listdir(self.parquet_dir)
            if f.endswith(".parquet")
        )
        if not files:
            raise FileNotFoundError(f"no .parquet files under {self.parquet_dir}")

        captions: List[str] = []
        images_raw: List = []
        for f in files:
            t = pq.read_table(f, columns=["image", "caption"])
            captions.extend(t.column("caption").to_pylist())
            images_raw.extend(t.column("image").to_pylist())

        n_total = len(captions)
        if n_total < self.num_samples:
            print(
                f"[coco_fid] WARNING: requested num_samples={self.num_samples} "
                f"> available={n_total}; using all {n_total}"
            )
            self.num_samples = n_total

        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(n_total)[: self.num_samples]
        self._captions = [captions[int(i)] for i in idx]
        self._gt_images = [
            _pil_short_resize_center_crop(
                _decode_coco_image(images_raw[int(i)]), self.image_size
            )
            for i in idx
        ]

    def _load_inception(self, device, dtype):
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        net = InceptionV3([block_idx])
        net.to(device, dtype=dtype).eval()
        net.requires_grad_(False)
        self._inception = net

    def _load_clip(self, device):
        """Load a CLIP model + processor (fp32, frozen) for CLIPScore."""
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(self.clip_model_path)
        model.to(device).eval()
        model.requires_grad_(False)
        self._clip_model = model
        self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_path)

    def setup(self, device, dtype):
        self._device = device
        self._dtype = dtype
        if self._captions is None:
            self._load_coco_subset()
        if self._inception is None:
            self._load_inception(device, dtype)
        if self.clip_model_path and self._clip_model is None:
            self._load_clip(device)

    # ------------------------------------------------------------------
    # Sharding
    # ------------------------------------------------------------------
    def shard_for_rank(self, rank: int, world_size: int) -> List[int]:
        n = len(self._captions) if self._captions is not None else self.num_samples
        chunk = math.ceil(n / world_size)
        start = rank * chunk
        return [(start + i) % n for i in range(chunk)]

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_features(self, pils: List[Image.Image]) -> np.ndarray:
        assert self._inception is not None, "call setup() first"
        device, dtype = self._device, self._dtype
        out = []
        bs = max(1, self.inception_batch_size)
        for i in range(0, len(pils), bs):
            batch = pils[i : i + bs]
            x = _pils_to_tensor_01(batch, device, dtype)  # [B, 3, H, W] in [0, 1]
            feat = self._inception(x)[0]                  # [B, 2048, 1, 1]
            feat = feat.squeeze(-1).squeeze(-1)           # [B, 2048]
            out.append(feat.float().cpu().numpy())
        if not out:
            return np.zeros((0, 2048), dtype=np.float32)
        return np.concatenate(out, axis=0).astype(np.float32)

    def _ensure_local_ref_features(self, rank: int, world_size: int):
        if self._ref_feats_local is not None:
            return
        idx = self.shard_for_rank(rank, world_size)
        gt = [self._gt_images[i] for i in idx]
        self._ref_feats_local = self.extract_features(gt)

    @torch.no_grad()
    def clip_score_local(self, pils: List[Image.Image], captions: List[str]) -> Tuple[float, int]:
        """Return (sum_cosine, count) of CLIP image-text cosine over a shard.

        Cosine is computed between the CLIP image embedding of each generated
        PIL and the CLIP text embedding of its caption. The caller turns the
        all-reduced ``sum / count`` into the reported CLIPScore (x100).
        """
        assert self._clip_model is not None, "CLIP not loaded"
        device = self._device
        model, processor = self._clip_model, self._clip_processor
        total = 0.0
        n = 0
        bs = max(1, self.clip_batch_size)
        for i in range(0, len(pils), bs):
            bp = pils[i : i + bs]
            bc = captions[i : i + bs]
            inputs = processor(
                text=bc, images=bp, return_tensors="pt",
                padding=True, truncation=True, max_length=77,
            ).to(device)
            out = model(**inputs)
            img_emb = F.normalize(out.image_embeds.float(), dim=-1)
            txt_emb = F.normalize(out.text_embeds.float(), dim=-1)
            sims = (img_emb * txt_emb).sum(dim=-1)  # cosine in [-1, 1]
            total += float(sims.sum().item())
            n += int(sims.shape[0])
        return total, n

    # ------------------------------------------------------------------
    # Top-level eval
    # ------------------------------------------------------------------
    def evaluate(
        self,
        generate_pil_fn: Callable[[str], Image.Image],
        rank: int,
        world_size: int,
        verbose: bool = True,
    ) -> Tuple[float, dict]:
        """Run the full FID pipeline.

        ``generate_pil_fn(caption_str) -> PIL.Image`` is called exactly
        ``chunk = ceil(N / world_size)`` times on this rank. Returns
        ``(fid, info)`` on rank 0; other ranks get ``(0.0, {...})``.
        """
        assert self._inception is not None, "call setup() first"
        idx = self.shard_for_rank(rank, world_size)
        n_total = world_size * len(idx)
        info = {
            "n_samples_total": n_total,
            "n_per_rank": len(idx),
            "image_size": self.image_size,
        }

        gen_pils: List[Image.Image] = []
        n_local = len(idx)
        log_every = max(1, n_local // 10)
        t_gen_start = time.perf_counter()
        for k, i in enumerate(idx):
            cap = self._captions[i]
            pil = generate_pil_fn(cap)
            pil = _pil_short_resize_center_crop(pil, self.image_size)
            gen_pils.append(pil)
            done = k + 1
            if verbose and rank == 0 and done % log_every == 0:
                elapsed = time.perf_counter() - t_gen_start
                avg = elapsed / done
                remaining = avg * (n_local - done)
                print(
                    f"[coco_fid] rank0 generated {done}/{n_local} "
                    f"(global ~{done * world_size}/{n_total}) | "
                    f"elapsed={_fmt_dur(elapsed)} "
                    f"avg={avg:.2f}s/img "
                    f"ETA={_fmt_dur(remaining)}"
                )

        if verbose and rank == 0:
            t_gen = time.perf_counter() - t_gen_start
            print(
                f"[coco_fid] generation done: {n_local} images on rank0 "
                f"in {_fmt_dur(t_gen)} ({t_gen / max(1, n_local):.2f}s/img)"
            )

        # ---- CLIPScore (image-text alignment of generated samples) ----
        # Computed over the SAME generated PILs + their captions, then
        # all-reduced (sum, count) across ranks -> mean cosine x100.
        clip_score = None
        if self._clip_model is not None:
            shard_caps = [self._captions[i] for i in idx]
            clip_sum_local, clip_cnt_local = self.clip_score_local(gen_pils, shard_caps)
            if world_size > 1:
                t = torch.tensor(
                    [clip_sum_local, float(clip_cnt_local)],
                    device=torch.device(f"cuda:{torch.cuda.current_device()}"),
                    dtype=torch.float64,
                )
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                clip_sum, clip_cnt = float(t[0].item()), float(t[1].item())
            else:
                clip_sum, clip_cnt = clip_sum_local, float(clip_cnt_local)
            clip_score = 100.0 * (clip_sum / max(clip_cnt, 1.0))
            info["clip_score"] = clip_score

        t_feat_start = time.perf_counter()
        self._ensure_local_ref_features(rank, world_size)
        gen_feats_local = self.extract_features(gen_pils)
        ref_feats_local = self._ref_feats_local

        if verbose and rank == 0:
            t_feat = time.perf_counter() - t_feat_start
            print(
                f"[coco_fid] InceptionV3 features extracted in "
                f"{_fmt_dur(t_feat)} | shapes: "
                f"local_gen={gen_feats_local.shape} local_ref={ref_feats_local.shape}; "
                f"all-gathering across world_size={world_size}..."
            )

        if world_size > 1:
            gen_feats_all = _all_gather_np_2d(gen_feats_local)
            ref_feats_all = _all_gather_np_2d(ref_feats_local)
        else:
            gen_feats_all = gen_feats_local
            ref_feats_all = ref_feats_local

        if rank != 0:
            return 0.0, {**info, "rank": rank, "skipped_log": True}

        info["gen_feats_shape"] = list(gen_feats_all.shape)
        info["ref_feats_shape"] = list(ref_feats_all.shape)

        mu_ref, mu_gen = ref_feats_all.mean(0), gen_feats_all.mean(0)
        sigma_ref = np.cov(ref_feats_all, rowvar=False)
        sigma_gen = np.cov(gen_feats_all, rowvar=False)
        fid = float(calculate_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen))
        info["fid"] = fid
        return fid, info
