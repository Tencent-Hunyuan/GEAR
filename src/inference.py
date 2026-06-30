"""Stand-alone sampling + NPZ packing for ``src`` AR + VQ checkpoints.

Rebuilds the AR + VQ models from a training checkpoint, runs distributed
class-conditional sampling (``generate(...) -> vq.decode_code(...)``),
saves the PNGs and packs them into a single ``samples.npz`` that
``tools/evaluator.py`` consumes for the **paper-quality** ADM-style metrics
(gFID / sFID / IS / Precision / Recall).

This script does **not** compute FID itself -- the in-process FID flow in
``src/train_gear.py`` (``run_distributed_fid_eval``) is the right
choice during training, but for the paper numbers we always go through ADM
on a saved NPZ. Run::

    # Generate + pack
    accelerate launch --num_processes 8 src/inference.py \\
        --ckpt-path /path/to/checkpoints/0400000.pt \\
        --output-dir /path/to/infer_out \\
        --fid-num 50000 \\
        --cfg-scale 1.0 \\
        --per-proc-batch-size 32

    # Then, from the ADM evaluator env (separate; see top-level README.md):
    python tools/evaluator.py \\
        /path/to/VIRTUAL_imagenet256_labeled.npz \\
        /path/to/infer_out/<run-tag>/samples.npz

Checkpoint shape recap:

* **Stage-1 ckpt** (``train_gear.py``): contains ``ar`` / ``ema`` /
  ``vq`` / ``vq_ema`` plus the saved ``args`` dict -- this script needs
  *only* ``--ckpt-path`` for the full AR+VQ stack.

* **Stage-2 ckpt** (``train_ar.py``): contains only ``ar`` / ``ema``
  (VQ is frozen and loaded from a stage-0 / stage-1 file at train time).
  Pass ``--vq-ckpt-path`` here (or it falls back to ``ck['args']['vq_ckpt']``).

EMA weights are used by default for both AR and VQ -- consistent with what
the training-time online FID uses for the wandb numbers.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Heavy imports come from the same place train_gear uses, so we always
# stay in sync with the trainer's model factory.
# Unified tokenizer registry (holds VQ / LFQ / IBQ families). Aliased to
# ``VQ_models`` so the existing ``VQ_models[args.vq_model]`` lookup keeps
# working for --vq-model=LFQ-16 / IBQ-16, not just VQ-8 / VQ-16.
from models import Tokenizers as VQ_models  # noqa: E402
from models.generate import generate  # noqa: E402
from models.llamagen import LlamaGen_models  # noqa: E402
from src.utils import load_pretrained_tokenizer_state_dict  # noqa: E402
from tools.save_npz import create_npz_from_sample_folder  # noqa: E402


# =============================================================================
# Checkpoint plumbing
# =============================================================================
def load_ckpt(ckpt_path: Path, *, map_location: str = "cpu") -> dict:
    """Load a training checkpoint and return its dict.

    Local training ckpts store the training-time args under ``"args"`` and we
    use it to rebuild the AR / VQ architectures automatically. Published
    (HuggingFace) weights have that ``args`` snapshot stripped, so it may be
    absent -- in that case we return an empty ``args`` dict and the caller is
    expected to supply the architecture via CLI flags (``--ar-model`` etc.),
    falling back to the canonical VQ-16 defaults.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=map_location)
    ck.setdefault("args", {})
    return ck


def _overlay_cli_overrides(args_dict: dict, args: argparse.Namespace) -> dict:
    """Overlay user CLI model-config flags onto the ckpt args dict.

    Precedence is therefore: CLI flag (if provided) > ckpt ``args`` (if present)
    > the canonical default baked into each ``_arg(..., default)`` call below.
    This lets published weights (whose ``args`` snapshot was stripped on upload)
    run by passing ``--ar-model`` / ``--image-size`` etc., while local training
    ckpts keep working with no extra flags.
    """
    overrides = {
        "ar_model": getattr(args, "ar_model", None),
        "vq_model": getattr(args, "vq_model", None),
        "codebook_size": getattr(args, "codebook_size", None),
        "codebook_embed_dim": getattr(args, "codebook_embed_dim", None),
        "downsample_ratio": getattr(args, "downsample_ratio", None),
        "num_classes": getattr(args, "num_classes", None),
        "cls_token_num": getattr(args, "cls_token_num", None),
        "image_size": getattr(args, "image_size", None),
    }
    for key, val in overrides.items():
        if val is not None:
            args_dict[key] = val
    return args_dict


def _arg(args_dict: dict, key: str, default=None):
    """Lookup an arg with sensible defaults for older ckpts that lack a field."""
    return args_dict.get(key, default)


# =============================================================================
# Model construction
# =============================================================================
def build_vq(args_dict: dict, device: torch.device) -> torch.nn.Module:
    """Reconstruct the VQ model exactly as ``train_gear.py`` does."""
    vq = VQ_models[_arg(args_dict, "vq_model", "VQ-16")](
        codebook_size=_arg(args_dict, "codebook_size", 16384),
        codebook_embed_dim=_arg(args_dict, "codebook_embed_dim", 8),
    )
    return vq.to(device).eval()


def build_ar(args_dict: dict, latent_size: int, device: torch.device) -> torch.nn.Module:
    """Reconstruct the AR model exactly as ``train_gear.py`` does.

    During *sampling* the dropout terms don't matter (no training), but we
    still mirror the trainer's signature so the state_dict load is strict.
    """
    block_size = latent_size ** 2
    ar = LlamaGen_models[_arg(args_dict, "ar_model", "LlamaGen-L")](
        block_size=block_size,
        vocab_size=_arg(args_dict, "codebook_size", 16384),
        num_classes=_arg(args_dict, "num_classes", 1000),
        cls_token_num=_arg(args_dict, "cls_token_num", 1),
        resid_dropout_p=_arg(args_dict, "dropout_p", 0.1),
        ffn_dropout_p=_arg(args_dict, "dropout_p", 0.1),
        token_dropout_p=_arg(args_dict, "token_dropout_p", 0.1),
        drop_path_rate=_arg(args_dict, "drop_path_rate", 0.0),
        use_checkpoint=False,  # checkpointing is a train-only memory trick
    )
    return ar.to(device).eval()


def _strip_compile_prefix(sd: dict) -> dict:
    """Remove ``_orig_mod.`` prefixes that torch.compile leaves behind."""
    if not sd or not next(iter(sd)).startswith("_orig_mod."):
        return sd
    return {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}


def load_ar_into(
    ar: torch.nn.Module,
    ckpt: dict,
    *,
    use_ema: bool,
    log_fn,
) -> None:
    """Load AR weights from a training ckpt. Mirrors trainer's save convention."""
    if use_ema and "ema" in ckpt:
        sd = ckpt["ema"]
        source = "ema"
    elif "ar" in ckpt:
        sd = ckpt["ar"]
        source = "ar (live)"
    elif "ema" in ckpt:
        sd = ckpt["ema"]
        source = "ema (no live in ckpt)"
    elif "model" in ckpt:
        # Legacy ckpts (top-level inference.py format).
        sd = ckpt["model"]
        source = "model (legacy)"
    else:
        raise KeyError(
            "ckpt has none of {'ema', 'ar', 'model'}; cannot load AR weights."
        )
    missing, unexpected = ar.load_state_dict(_strip_compile_prefix(sd), strict=False)
    log_fn(f"[AR ] loaded '{source}'  missing={len(missing)}  unexpected={len(unexpected)}")


def load_vq_into(
    vq: torch.nn.Module,
    ckpt: dict,
    *,
    ckpt_path: Path,
    use_ema: bool,
    explicit_vq_ckpt: Path | None,
    log_fn,
) -> None:
    """Load VQ weights.

    Priority:

    1. ``--vq-ckpt-path`` if explicitly passed (always wins).
    2. ``ckpt['vq_ema']`` / ``ckpt['vq']`` for stage1 ckpts.
    3. Stage2: look up ``args.vq_ckpt`` in the saved-args dict and use
       :func:`load_pretrained_tokenizer_state_dict` to extract the right
       sub-dict (handles MAGVIT2 / stage0 init formats).
    """
    if explicit_vq_ckpt is not None:
        sd = load_pretrained_tokenizer_state_dict(
            str(explicit_vq_ckpt),
            use_ema=use_ema,
        )
        source = f"--vq-ckpt-path {explicit_vq_ckpt} (use_ema={use_ema})"
    elif use_ema and "vq_ema" in ckpt:
        sd = ckpt["vq_ema"]
        source = "ckpt['vq_ema']"
    elif "vq" in ckpt:
        sd = ckpt["vq"]
        source = "ckpt['vq']"
    elif "vq_ema" in ckpt:
        sd = ckpt["vq_ema"]
        source = "ckpt['vq_ema'] (no live vq in ckpt)"
    else:
        # Stage-2 fallback: trainer saved `args.vq_ckpt` -- reuse it.
        saved_vq_ckpt = _arg(ckpt["args"], "vq_ckpt")
        if not saved_vq_ckpt:
            raise KeyError(
                f"ckpt {ckpt_path} contains no 'vq' / 'vq_ema' keys (looks like "
                f"a stage-2 ckpt) and ck['args']['vq_ckpt'] is missing too. "
                f"Re-run with --vq-ckpt-path pointing at the stage-1 / stage-0 ckpt."
            )
        sd = load_pretrained_tokenizer_state_dict(
            str(saved_vq_ckpt),
            use_ema=use_ema,
        )
        source = f"ck['args']['vq_ckpt'] = {saved_vq_ckpt} (use_ema={use_ema})"

    missing, unexpected = vq.load_state_dict(_strip_compile_prefix(sd), strict=False)
    log_fn(f"[VQ ] loaded {source}  missing={len(missing)}  unexpected={len(unexpected)}")


# =============================================================================
# Sampling loop (mirrors run_distributed_fid_eval, sans the InceptionV3 part)
# =============================================================================
def _sampling_tag(args: argparse.Namespace) -> str:
    """Filesystem-safe tag describing the sampling config (cfg / temp / topk / ...).

    This is the *leaf* of the 3-level output hierarchy
    ``<output-dir>/<exp-name>/<step-tag>/<sampling-tag>/``, so different
    sampling sweeps over the same checkpoint live side by side.
    """
    parts = [f"cfg{args.cfg_scale:g}", f"temp{args.temperature:g}"]
    if args.cfg_interval >= 0:
        parts.insert(1, f"cfgint{args.cfg_interval:g}")
    if args.top_k > 0:
        parts.append(f"topk{args.top_k}")
    if args.top_p < 1.0:
        parts.append(f"topp{args.top_p:g}")
    if args.tag:
        parts.append(args.tag)
    return "-".join(parts)


def _resolve_exp_name(ckpt_path: Path, override: str | None) -> str:
    """Pick a sensible experiment name for the output sub-folder.

    Trainer layout is ``<exp_root>/<exp_name>/checkpoints/<step>.pt``, so the
    grandparent of the ckpt is the exp name. Falls back to the immediate
    parent's name when the layout is non-standard. ``--exp-name`` always wins.
    """
    if override:
        return override
    if ckpt_path.parent.name == "checkpoints":
        return ckpt_path.parent.parent.name
    return ckpt_path.parent.name


def _step_tag(ckpt_path: Path) -> str:
    """e.g. ``0400000.pt`` -> ``step0400000``."""
    return f"step{ckpt_path.stem}"


def _make_log_fn(accelerator: Accelerator):
    """Rank-aware print helper."""
    def log(*msgs):
        if accelerator.is_main_process:
            print("[infer]", *msgs, flush=True)
    return log


@torch.no_grad()
def sample_and_save(
    accelerator: Accelerator,
    ar: torch.nn.Module,
    vq: torch.nn.Module,
    *,
    args: argparse.Namespace,
    args_dict: dict,
    sample_dir: Path,
    log_fn,
) -> None:
    """Distributed sampling -> PNGs on disk.

    Layout mirrors ``run_distributed_fid_eval`` so PNGs cover
    ``[0, total_samples)`` contiguously across ranks.
    """
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    image_size = _arg(args_dict, "image_size", 256)
    downsample = _arg(args_dict, "downsample_ratio", 16)
    latent_size = image_size // downsample
    num_classes = _arg(args_dict, "num_classes", 1000)
    codebook_embed_dim = _arg(args_dict, "codebook_embed_dim", 8)

    n = int(args.per_proc_batch_size)
    global_batch = n * world_size
    total_samples = int(math.ceil(args.fid_num / global_batch) * global_batch)
    iters = total_samples // global_batch

    log_fn(
        f"image_size={image_size} latent_size={latent_size} "
        f"num_classes={num_classes} codebook_embed_dim={codebook_embed_dim}"
    )
    log_fn(
        f"world_size={world_size} per_proc_batch_size={n} "
        f"global_batch={global_batch} fid_num={args.fid_num} "
        f"-> total_samples={total_samples} iters={iters}"
    )
    log_fn(
        f"sampling: cfg_scale={args.cfg_scale} cfg_interval={args.cfg_interval} "
        f"temperature={args.temperature} top_k={args.top_k} top_p={args.top_p} "
        f"seed={args.seed}"
    )

    if accelerator.is_main_process:
        sample_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    # Resumability: skip iters whose PNGs already exist.
    existing = {p.stem for p in sample_dir.glob("*.png")}
    if accelerator.is_main_process and existing:
        log_fn(f"resume: found {len(existing)} PNGs in {sample_dir}, will skip them.")

    # Each rank uses its own seed offset so generated images are different
    # across ranks even for the same iter index.
    g = torch.Generator(device=device)
    g.manual_seed(int(args.seed) * world_size + rank)

    pbar = tqdm(range(iters), disable=not accelerator.is_main_process,
                desc="sample", unit="iter")
    total = 0
    t0 = time.time()
    for _ in pbar:
        c_indices = torch.randint(0, int(num_classes), (n,), device=device, generator=g)
        qzshape = [n, int(codebook_embed_dim), latent_size, latent_size]

        # `generate` lives in models/generate.py and does the AR rollout with
        # CFG + top-k/top-p, mirroring train_gear's online sample path.
        index_sample = generate(
            ar, c_indices, latent_size ** 2,
            cfg_scale=args.cfg_scale,
            cfg_interval=args.cfg_interval,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_logits=True,
        )
        # decode_code returns float images in [-1, 1].
        samples = vq.decode_code(index_sample, qzshape)
        if args.image_size_eval != image_size:
            samples = torch.nn.functional.interpolate(
                samples, size=(args.image_size_eval, args.image_size_eval),
                mode="bicubic",
            )
        samples = (
            torch.clamp(127.5 * samples + 128.0, 0, 255)
            .permute(0, 2, 3, 1)
            .to("cpu", dtype=torch.uint8)
            .numpy()
        )

        for i in range(samples.shape[0]):
            index = i * world_size + rank + total
            if index >= args.fid_num:
                continue
            fname = f"{index:06d}"
            if fname in existing:
                continue
            Image.fromarray(np.ascontiguousarray(samples[i])).save(
                str(sample_dir / f"{fname}.png")
            )
        total += global_batch

    accelerator.wait_for_everyone()
    log_fn(f"sampling done in {time.time() - t0:.1f}s -> {sample_dir}")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ---- checkpoint -----------------------------------------------------------
    p.add_argument(
        "--ckpt-path", type=Path, required=True,
        help="Path to a training checkpoint .pt from train_gear.py or "
             "train_ar.py.",
    )
    p.add_argument(
        "--vq-ckpt-path", type=Path, default=None,
        help="Optional override for the VQ weights. Needed when --ckpt-path "
             "is a stage2 checkpoint that doesn't bundle 'vq'/'vq_ema' keys "
             "and ck['args']['vq_ckpt'] isn't reachable on this machine.",
    )
    p.add_argument(
        "--use-ar-ema", default=True,
        action=argparse.BooleanOptionalAction,
        help="Use AR EMA weights (default; matches the paper-style FID "
             "evaluation). Pass --no-use-ar-ema for the live AR.",
    )
    p.add_argument(
        "--use-vq-ema", default=True,
        action=argparse.BooleanOptionalAction,
        help="Use VQ EMA weights (default). Pass --no-use-vq-ema for the live VQ.",
    )

    # ---- model architecture (override ckpt args; REQUIRED for published HF
    #      weights, whose args snapshot is stripped on upload) -----------------
    # Each defaults to None => fall back to the ckpt's saved args, then to the
    # canonical VQ-16 default. Pass --ar-model (and --image-size for 512 models)
    # when evaluating downloaded weights.
    p.add_argument("--ar-model", type=str, default=None,
                   help="AR family, e.g. LlamaGen-B/L/XL. Required for HF weights.")
    p.add_argument("--image-size", type=int, default=None,
                   help="Generation resolution the AR was trained at (256/384/512).")
    p.add_argument("--vq-model", type=str, default=None,
                   help="Tokenizer family (default VQ-16).")
    p.add_argument("--codebook-size", type=int, default=None,
                   help="Codebook size (default 16384).")
    p.add_argument("--codebook-embed-dim", type=int, default=None,
                   help="Codebook embedding dim (VQ=8, LFQ=14, IBQ=256).")
    p.add_argument("--downsample-ratio", type=int, default=None,
                   help="VQ spatial downsample ratio (default 16).")
    p.add_argument("--num-classes", type=int, default=None,
                   help="Number of conditioning classes (default 1000).")
    p.add_argument("--cls-token-num", type=int, default=None,
                   help="Class-token prefix length (default 1 for c2i).")

    # ---- sampling -------------------------------------------------------------
    p.add_argument("--fid-num", type=int, default=50000,
                   help="Total number of images to sample (default: 50000).")
    p.add_argument("--per-proc-batch-size", type=int, default=32,
                   help="Per-rank batch size during sampling.")
    p.add_argument("--cfg-scale", type=float, default=1.0,
                   help="Classifier-free guidance scale (1.0 = no CFG, paper-style).")
    p.add_argument("--cfg-interval", type=float, default=-1,
                   help="CFG interval (-1 = always-on across the AR rollout).")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature.")
    p.add_argument("--top-k", type=int, default=0,
                   help="top-k filter (0 = disabled).")
    p.add_argument("--top-p", type=float, default=1.0,
                   help="top-p (nucleus) filter (1.0 = disabled).")
    p.add_argument("--seed", type=int, default=0,
                   help="Base RNG seed. Per-rank seed is seed*world_size + rank.")

    # ---- output ---------------------------------------------------------------
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Parent directory; the script writes samples + npz into "
                        "<output-dir>/<exp-name>/<step-tag>/<sampling-tag>/, so "
                        "different ckpt steps and sampling sweeps of the same "
                        "experiment cohabit cleanly.")
    p.add_argument("--exp-name", type=str, default="",
                   help="Override for the experiment-name sub-folder. By "
                        "default we use the grandparent of the ckpt "
                        "(i.e. <exp_root>/<exp_name>/checkpoints/<step>.pt -> "
                        "<exp_name>).")
    p.add_argument("--tag", type=str, default="",
                   help="Optional suffix appended to the sampling-tag leaf "
                        "(e.g. 'topp0p95-rerun-seed1').")
    p.add_argument("--image-size-eval", type=int, default=256,
                   help="Resize generated images to this size before saving "
                        "(default: use the trained --image-size). Useful when "
                        "comparing against a reference NPZ at a different res.")
    p.add_argument("--keep-pngs", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="Keep the per-image PNGs after the NPZ is packed "
                        "(default: keep). Pass --no-keep-pngs to delete the "
                        "PNG folder once samples.npz is written.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    accelerator = Accelerator()
    log_fn = _make_log_fn(accelerator)
    device = accelerator.device

    # Determinism for replays + per-rank generator (see sample_and_save).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    torch.manual_seed(int(args.seed) + accelerator.process_index)

    log_fn(f"world_size={accelerator.num_processes} device={device}")
    log_fn(f"loading ckpt: {args.ckpt_path}")

    # CPU-load once; rank-local move-to-GPU happens after model build.
    ck = load_ckpt(args.ckpt_path, map_location="cpu")
    had_saved_args = bool(ck["args"])
    args_dict = _overlay_cli_overrides(ck["args"], args)
    if not had_saved_args:
        log_fn("ckpt has no saved 'args' (published weights); using CLI flags "
               "+ canonical VQ-16 defaults for the architecture.")
    if args.image_size_eval is None:
        args.image_size_eval = _arg(args_dict, "image_size", 256)
    log_fn(
        f"ckpt args: ar={_arg(args_dict, 'ar_model')} "
        f"vq={_arg(args_dict, 'vq_model')} "
        f"image_size={_arg(args_dict, 'image_size')} "
        f"codebook_size={_arg(args_dict, 'codebook_size')} "
        f"steps={ck.get('steps', '?')}"
    )

    image_size = _arg(args_dict, "image_size", 256)
    downsample = _arg(args_dict, "downsample_ratio", 16)
    latent_size = image_size // downsample

    vq = build_vq(args_dict, device)
    ar = build_ar(args_dict, latent_size, device)

    load_vq_into(
        vq, ck,
        ckpt_path=args.ckpt_path,
        use_ema=args.use_vq_ema,
        explicit_vq_ckpt=args.vq_ckpt_path,
        log_fn=log_fn,
    )
    load_ar_into(ar, ck, use_ema=args.use_ar_ema, log_fn=log_fn)

    # Free the on-CPU ckpt dict before sampling (50k samples + 8 GPUs leaves
    # very little spare host RAM; the AR+VQ weights are already on GPU).
    del ck

    exp_name = _resolve_exp_name(args.ckpt_path, args.exp_name)
    step_tag = _step_tag(args.ckpt_path)
    sampling_tag = _sampling_tag(args)
    run_dir = args.output_dir / exp_name / step_tag / sampling_tag
    sample_dir = run_dir / "samples"
    npz_path = run_dir / "samples.npz"
    log_fn(f"run_dir={run_dir}")
    log_fn(f"  exp_name={exp_name}  step_tag={step_tag}  sampling_tag={sampling_tag}")

    # If the user already produced this NPZ, no point sampling again.
    if accelerator.is_main_process and npz_path.exists():
        log_fn(f"FOUND existing {npz_path}; nothing to do. "
               f"Delete it (or pass a different --tag) to re-sample.")
    accelerator.wait_for_everyone()
    if npz_path.exists():
        return

    sample_and_save(
        accelerator, ar, vq,
        args=args,
        args_dict=args_dict,
        sample_dir=sample_dir,
        log_fn=log_fn,
    )

    if accelerator.is_main_process:
        log_fn(f"packing NPZ: {npz_path}")
        npz_made = create_npz_from_sample_folder(str(sample_dir), int(args.fid_num))
        # create_npz_from_sample_folder writes ``<sample_dir>.npz``; rename to
        # our canonical samples.npz so the consumer command is stable.
        if Path(npz_made) != npz_path:
            shutil.move(str(npz_made), str(npz_path))
        log_fn(f"NPZ ready: {npz_path}")

        if not args.keep_pngs:
            try:
                shutil.rmtree(sample_dir)
                log_fn(f"deleted PNG folder {sample_dir}")
            except OSError as exc:
                log_fn(f"WARN failed to clean {sample_dir}: {exc}")

        ref_hint = "/path/to/VIRTUAL_imagenet256_labeled.npz"
        log_fn(
            "\n  --- next step: ADM evaluator (separate dit_eval env) ---\n"
            f"    python tools/evaluator.py \\\n"
            f"        {ref_hint} \\\n"
            f"        {npz_path}\n"
        )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
