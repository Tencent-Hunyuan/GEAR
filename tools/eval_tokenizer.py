"""Standalone tokenizer eval -- one script for VQ-16 / LFQ-16 / (future IBQ-*).

Why this exists
---------------
``train_tokenizer/gear.py`` runs the same eval inline every ``--vq-eval-steps``
steps as part of training. This script lifts that *exact* eval out into a
one-shot CLI so we can:

  * sanity-check a freshly-trained tokenizer ckpt without spinning up the
    full training loop again;
  * compare a Stage-0 / Stage-1 ckpt against a public reference ckpt
    (e.g. ``vq_ds16_c2i.pt`` or the TencentARC MAGVIT2 pretrain) on the
    same val split with the same metric definitions;
  * mix-and-match: any combination of ``--vq-model`` registered in
    ``models.Tokenizers`` and any checkpoint we know how to read.

Recipe
------
We reuse :func:`src.utils.run_vq_reconstruction_eval` verbatim, so
``vq_val/{l1, psnr, ssim, fid}`` here are byte-identical to the values
reported under the same keys during training. See the docstring of
``run_vq_reconstruction_eval`` for the per-metric semantics.

Checkpoint formats supported
----------------------------
The ``--ckpt`` flag autodetects the layout:

  * ``{"vq": state_dict, ...}`` -- our Stage-0 / Stage-1 ckpts.
  * ``{"vq_ema": state_dict, ...}`` -- pass ``--use-ema`` to load this
    one. Falls back to ``"vq"`` if EMA is absent.
  * ``{"model": state_dict}``    -- legacy LlamaGen ``vq_ds16_c2i.pt``.
  * ``{"state_dict": {...model_ema.*...}}`` -- TencentARC MAGVIT2
    Lightning ckpt. Routed through
    :func:`models.lfq_model.convert_magvit2_pretrained_state_dict`
    which prefers the EMA shadow by default (override with
    ``--no-prefer-magvit2-ema``).
  * Bare state_dict (no wrapping)             -- accepted as-is.

Launching
---------
Single-GPU::

    python3 tools/eval_tokenizer.py \\
        --vq-model VQ-16 --codebook-size 16384 --codebook-embed-dim 8 \\
        --ckpt /path/to/vq_ds16_c2i.pt

Multi-GPU (faster on 50K val)::

    accelerate launch --num_processes=8 tools/eval_tokenizer.py \\
        --vq-model LFQ-16 --codebook-size 16384 --codebook-embed-dim 14 \\
        --ckpt /path/to/pretrain256_16384.ckpt

Multi-node::

    torchrun --nnodes=N --nproc_per_node=8 --master_addr=... ... \\
        tools/eval_tokenizer.py [args]

``accelerate`` / ``torchrun`` env vars are honoured automatically;
single-GPU launches without them just run on rank 0.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from models import Tokenizers  # noqa: E402
from models.lfq_model import convert_magvit2_pretrained_state_dict  # noqa: E402

from src.dataset import build_imagenet_val_dataset  # noqa: E402
from src.utils import run_vq_reconstruction_eval  # noqa: E402


def _load_tokenizer_state_dict(
    ckpt_path: str,
    use_ema: bool,
    prefer_magvit2_ema: bool,
    log: logging.Logger,
):
    """Best-effort state_dict extraction across the formats we ship.

    Returns the raw state_dict (no module wrapping); caller does the
    ``load_state_dict(..., strict=False)`` and the missing/unexpected
    diagnostics.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 1) MAGVIT2 lightning format: has top-level "state_dict" containing
    #    both live and EMA params. Detect via the LitEma "model_ema.*" prefix.
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        inner = raw["state_dict"]
        if any(k.startswith("model_ema.") for k in inner):
            # IBQ ships a learnable codebook (``quantize.embedding.*``) plus
            # ``quant_conv`` / ``post_quant_conv``; the single-codebook
            # MAGVIT2 LFQ ckpt has none of these. Route IBQ ckpts through
            # the IBQ converter, everything else through the MAGVIT2 one.
            if any("quantize.embedding" in k for k in inner):
                from models.ibq_model import convert_ibq_pretrained_state_dict  # noqa: E402

                log.info(
                    f"[ckpt] detected IBQ lightning format "
                    f"(prefer_ema={prefer_magvit2_ema})"
                )
                return convert_ibq_pretrained_state_dict(inner, prefer_ema=prefer_magvit2_ema)
            log.info(
                f"[ckpt] detected MAGVIT2 lightning format "
                f"(prefer_ema={prefer_magvit2_ema})"
            )
            return convert_magvit2_pretrained_state_dict(inner, prefer_ema=prefer_magvit2_ema)
        # Lightning checkpoint without EMA shadow -- still flatten.
        log.info("[ckpt] detected lightning state_dict (no EMA shadow)")
        return inner

    # 2) Our Stage-0 / Stage-1 ckpts.
    if isinstance(raw, dict) and ("vq" in raw or "vq_ema" in raw):
        if use_ema and "vq_ema" in raw:
            log.info("[ckpt] using `vq_ema` (EMA shadow)")
            return raw["vq_ema"]
        if use_ema and "vq_ema" not in raw:
            log.info("[ckpt] --use-ema requested but `vq_ema` absent; falling back to `vq`")
        log.info("[ckpt] using `vq` (live params)")
        return raw["vq"]

    # 3) Legacy `{"model": ...}` wrapping (vq_ds16_c2i.pt and friends).
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        log.info("[ckpt] using legacy `model` key")
        return raw["model"]

    # 4) Bare state_dict.
    if isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
        log.info("[ckpt] using bare state_dict")
        return raw

    raise ValueError(
        f"Could not infer a state_dict from {ckpt_path}. Top-level keys: "
        f"{list(raw.keys()) if isinstance(raw, dict) else type(raw)}"
    )


def _make_logger(accelerator: Accelerator, out_dir: Path | None) -> logging.Logger:
    """Stream + optional file logger, rank-0-only."""
    log = logging.getLogger("eval_tokenizer")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    if not accelerator.is_main_process:
        log.disabled = True
        return log
    fmt = logging.Formatter(
        "[\033[34m%(asctime)s\033[0m] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(out_dir / "eval.log")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


def main(args):
    accelerator = Accelerator()
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    device = accelerator.device

    out_dir: Path | None = Path(args.out_dir) if args.out_dir else None
    log = _make_logger(accelerator, out_dir)

    # ---- 1. Build tokenizer from the unified registry ----------------------
    if args.vq_model not in Tokenizers:
        raise ValueError(
            f"Unknown tokenizer {args.vq_model!r}. Registered: {sorted(Tokenizers)}"
        )
    model_kwargs = dict(
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    )
    vq = Tokenizers[args.vq_model](**model_kwargs)

    log.info(
        f"Built {args.vq_model}: {sum(p.numel() for p in vq.parameters()):,} params "
        f"(codebook_size={vq.config.codebook_size}, "
        f"codebook_embed_dim={vq.config.codebook_embed_dim})"
    )

    # ---- 2. Load checkpoint -------------------------------------------------
    sd = _load_tokenizer_state_dict(
        args.ckpt,
        use_ema=args.use_ema,
        prefer_magvit2_ema=args.prefer_magvit2_ema,
        log=log,
    )
    missing, unexpected = vq.load_state_dict(sd, strict=False)
    log.info(
        f"[ckpt] loaded {args.ckpt} | missing={len(missing)}, unexpected={len(unexpected)}"
    )
    # Loud about missing trainable params; quiet about buffer-only misses
    # (LFQ's `quantize.mask` / `quantize.codebook` are non-persistent
    # buffers so they never appear in any saved ckpt).
    sub_missing = [k for k in missing if k.startswith(("encoder.", "decoder.", "quant_conv", "post_quant_conv", "quantize.embedding"))]
    if sub_missing:
        log.warning(f"[ckpt] missing trainable params (first 10): {sub_missing[:10]}")
    if unexpected:
        log.warning(f"[ckpt] unexpected (first 10): {unexpected[:10]}")

    vq = vq.to(device).eval()

    # ---- 3. Val dataloader (DistributedSampler -- mandatory contract for
    #    run_vq_reconstruction_eval; without it gather() crashes on uneven
    #    shards in multi-rank runs) ------------------------------------------
    val_dataset = build_imagenet_val_dataset(
        args.data_dir,
        image_size=args.image_size,
        resize_mode=args.val_resize_mode,
    )
    log.info(f"[data] val resize_mode={args.val_resize_mode!r}")
    if args.max_samples > 0 and args.max_samples < len(val_dataset):
        # Deterministic head-slice. Used for fast smoke tests; metrics are
        # still globally consistent because every rank slices the same head.
        from torch.utils.data import Subset
        val_dataset = Subset(val_dataset, list(range(args.max_samples)))
        log.info(f"[data] subsetting val to first {args.max_samples} samples")

    if accelerator.use_distributed:
        sampler = DistributedSampler(
            val_dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=False,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    log.info(
        f"[data] val set: {len(val_dataset):,} images "
        f"(per-rank batches: {len(val_loader)})"
    )

    # ---- 4. Optional InceptionV3 for FID -----------------------------------
    inception = None
    if args.eval_fid:
        from tools.calculate_fid import InceptionV3  # noqa: WPS433

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        inception = InceptionV3([block_idx]).to(device).eval()
        for p in inception.parameters():
            p.requires_grad = False
        log.info("[fid] loaded InceptionV3 (block 2048)")

    # ---- 5. Run eval (the same helper Stage 0/1 use inline) -----------------
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    metrics = run_vq_reconstruction_eval(
        accelerator=accelerator,
        vq=vq,
        val_loader=val_loader,
        inception=inception,
        eval_l1=args.eval_l1,
        eval_psnr=args.eval_psnr,
        eval_ssim=args.eval_ssim,
        log=log if accelerator.is_main_process else None,
    )

    # ---- 6. Persist results ------------------------------------------------
    if accelerator.is_main_process:
        log.info("=" * 60)
        log.info(f"Final metrics for {args.vq_model} @ {args.ckpt}:")
        for k, v in metrics.items():
            log.info(f"  {k:20s} = {v:.4f}")
        log.info("=" * 60)

        if out_dir is not None:
            payload = {
                "vq_model": args.vq_model,
                "ckpt": args.ckpt,
                "use_ema": args.use_ema,
                "prefer_magvit2_ema": args.prefer_magvit2_ema,
                "data_dir": args.data_dir,
                "image_size": args.image_size,
                "val_resize_mode": args.val_resize_mode,
                "num_samples": len(val_dataset),
                "batch_size": args.batch_size,
                "metrics": metrics,
            }
            with open(out_dir / "eval_metrics.json", "w") as f:
                json.dump(payload, f, indent=2)
            log.info(f"Wrote {out_dir / 'eval_metrics.json'}")

    accelerator.wait_for_everyone()


def parse_args():
    p = argparse.ArgumentParser()

    # Model
    p.add_argument(
        "--vq-model", type=str, required=True,
        help="Tokenizer family / size. Looked up in models.Tokenizers "
             "(currently: VQ-8 / VQ-16 / LFQ-16).",
    )
    p.add_argument(
        "--codebook-size", type=int, default=16384,
        help="Codebook size for the chosen tokenizer family.",
    )
    p.add_argument(
        "--codebook-embed-dim", type=int, default=8,
        help="Codebook embed dim. For LFQ this is forced to log2(codebook_size) "
             "regardless. Default 8 = the VQ-16 default; use 14 for LFQ-16.",
    )

    # Checkpoint
    p.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to a tokenizer checkpoint. See file docstring for the "
             "supported wrappings.",
    )
    p.add_argument(
        "--use-ema", action=argparse.BooleanOptionalAction, default=False,
        help="For Stage-0 / Stage-1 ckpts that carry both `vq` and `vq_ema`, "
             "use the EMA shadow. No effect on other ckpt formats.",
    )
    p.add_argument(
        "--prefer-magvit2-ema", action=argparse.BooleanOptionalAction, default=True,
        help="For TencentARC MAGVIT2 lightning ckpts (which always carry an "
             "EMA shadow), prefer the EMA copy of encoder/decoder. Matches "
             "the MAGVIT2 reference inference recipe.",
    )

    # Data
    p.add_argument(
        "--data-dir", type=str, required=True,
        help="ImageNet val root in `<root>/<synset>/<file>.JPEG` layout.",
    )
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument(
        "--val-resize-mode", type=str, default="bicubic",
        choices=["bicubic", "bilinear"],
        help="Short-side resize interpolation. Default 'bicubic' matches "
             "our train-time pipeline so train and eval see the same "
             "pixel distribution. Use 'bilinear' to match the SEED-Voken / "
             "Open-MAGVIT2 published numbers (their torchvision Resize "
             "default is BILINEAR). On the LFQ-16 MAGVIT2 pretrain the "
             "bicubic->bilinear swap moves PSNR by ~+0.7 dB and SSIM by "
             "~+0.03 just from changing the GT pixels.",
    )
    p.add_argument("--batch-size", type=int, default=32,
                   help="Per-rank batch size for the eval forward pass.")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--max-samples", type=int, default=0,
        help="Cap the val set to its first N samples (deterministic head "
             "slice). 0 = use the full 50k. Useful for fast smoke tests.",
    )

    # Metrics
    p.add_argument("--eval-l1", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-psnr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--eval-ssim", action=argparse.BooleanOptionalAction, default=True,
        help="SSIM via skimage. CPU-bound (~50-100ms/image); turn off for "
             "tight smoke tests if SSIM is not needed.",
    )
    p.add_argument(
        "--eval-fid", action=argparse.BooleanOptionalAction, default=True,
        help="Compute FID between (recon, ref) Inception features. Loads "
             "InceptionV3 on every rank.",
    )

    # Misc
    p.add_argument(
        "--out-dir", type=str, default=None,
        help="If set, dump eval_metrics.json + eval.log here.",
    )
    p.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    main(parse_args())
