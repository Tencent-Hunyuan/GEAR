"""Stand-alone text-to-image sampling + image/NPZ packing for ``src``
Stage-2 **t2i** checkpoints, purpose-built for the **GPIC FD-DINOv2** eval.

This is the t2i sibling of ``src/inference.py``. It is kept separate on
purpose: ``inference.py`` is hardwired for ImageNet class-conditional sampling
(``models.llamagen`` + ``models.generate`` with random ``c_indices`` and no
text encoder), whereas the GPIC eval is prompt-driven and needs the dual-stream
``models.llamagen_t2i`` AR, the frozen Qwen text encoder, and
``models.generate_t2i``.

Pipeline (mirrors the FID flow: sample -> save -> eval):

    1. Read a prompt JSONL (``{key, caption, caption_type}`` per line). Build it
       from the GPIC test reference with ``tools/build_gpic_prompts.py`` so the
       prompts are MATCHED to ``reference_stats/test_stats.npz`` (one image per
       reference key, from that key's own caption).
    2. Distributed sampling: each rank handles a disjoint slice of the prompt
       list, runs ``generate_t2i`` (default ``--cfg-scale 1.0`` = no CFG) and
       ``vq.decode_code``, writing one PNG per prompt.
    3. Pack the PNGs into a single ``samples.npz`` (``arr_0`` NHWC uint8) for
       convenience -- the gpic eval accepts EITHER the PNG directory or the npz.

Run::

    accelerate launch --num_processes 8 src/inference_t2i.py \\
        --ckpt-path /path/to/checkpoints/0400000.pt \\
        --prompts-jsonl /path/to/gpic/gpic_eval_50k.jsonl \\
        --output-dir /path/to/gpic_infer_out \\
        --per-proc-batch-size 32 \\
        --cfg-scale 1.0 --temperature 1.0

Then::

    python src/eval/gpic/gpic_eval_dino.py \\
        /path/to/gpic_infer_out/<exp>/<step>/<tag>/samples.npz \\
        /path/to/gpic/reference_stats/test_stats.npz \\
        --metrics fd,prdc,mmd

The VQ tokenizer is loaded the same way as ``inference.py``: explicit
``--vq-ckpt-path`` wins, else ``ckpt['vq'/'vq_ema']`` (rare for t2i), else the
trainer's saved ``ck['args']['vq_ckpt']``. EMA weights are used by default for
both AR and VQ, matching the training-time eval.
"""

from __future__ import annotations

import argparse
import json
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

from models import Tokenizers as VQ_models  # noqa: E402
from models.llamagen_t2i import LlamaGen_models  # noqa: E402
from models.generate_t2i import generate_t2i  # noqa: E402
from src.utils import load_pretrained_tokenizer_state_dict  # noqa: E402
from tools.save_npz import create_npz_from_sample_folder  # noqa: E402

from transformers import AutoModel, AutoTokenizer  # noqa: E402


# =============================================================================
# Checkpoint plumbing (shared shape with inference.py)
# =============================================================================
def load_ckpt(ckpt_path: Path, *, map_location: str = "cpu") -> dict:
    """Load a t2i checkpoint.

    Local training ckpts carry the training-time ``args`` snapshot (used to
    rebuild the AR / VQ stack automatically). Published (HuggingFace) weights
    have it stripped on upload, so it may be absent -- we then return an empty
    ``args`` dict and the caller supplies the architecture via CLI flags
    (``--ar-model`` / ``--image-size`` / ``--text-encoder`` ...), falling back
    to the canonical defaults.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=map_location)
    ck.setdefault("args", {})
    return ck


def _arg(args_dict: dict, key: str, default=None):
    return args_dict.get(key, default)


def _overlay_cli_overrides(args_dict: dict, args: argparse.Namespace) -> dict:
    """Overlay user CLI model-config flags onto the ckpt args dict.

    Precedence: CLI flag (if provided) > ckpt ``args`` (if present) > the
    canonical default baked into each ``_arg(..., default)`` call. Lets
    published t2i weights (args stripped on upload) run by passing
    ``--ar-model`` (+ ``--image-size 512`` for the 512 models).
    """
    overrides = {
        "ar_model": getattr(args, "ar_model", None),
        "vq_model": getattr(args, "vq_model", None),
        "codebook_size": getattr(args, "codebook_size", None),
        "codebook_embed_dim": getattr(args, "codebook_embed_dim", None),
        "downsample_ratio": getattr(args, "downsample_ratio", None),
        "cls_token_num": getattr(args, "cls_token_num", None),
        "text_attn": getattr(args, "text_attn", None),
        "image_size": getattr(args, "image_size", None),
    }
    for key, val in overrides.items():
        if val is not None:
            args_dict[key] = val
    return args_dict


def _strip_compile_prefix(sd: dict) -> dict:
    """Remove ``_orig_mod.`` prefixes that torch.compile leaves behind."""
    if not sd or not next(iter(sd)).startswith("_orig_mod."):
        return sd
    return {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}


# =============================================================================
# Frozen Qwen text encoder (copied from train_ar_t2i.py to stay standalone)
# =============================================================================
def load_text_encoder(path, device, dtype):
    """Load the frozen Qwen text backbone + tokenizer.

    ``AutoModel`` returns a model with ``.language_model`` (text) and possibly
    ``.visual`` (vision); we keep only the text tower. Returns
    ``(tokenizer, text_model, hidden_size)``.
    """
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    full = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=dtype)
    text_model = full.language_model if hasattr(full, "language_model") else full
    if hasattr(full, "visual"):
        del full.visual
    text_model = text_model.to(device).eval()
    for p in text_model.parameters():
        p.requires_grad = False
    hidden_size = text_model.config.hidden_size
    return tokenizer, text_model, hidden_size


@torch.no_grad()
def encode_text(text_model, input_ids, attn_mask):
    out = text_model(input_ids=input_ids, attention_mask=attn_mask)
    return out.last_hidden_state


@torch.no_grad()
def encode_prompts(tokenizer, text_model, prompts, text_max_len, device):
    """Tokenize + encode a list of prompts. Returns (emb (B,T,H), mask (B,T))."""
    enc = tokenizer(
        prompts, max_length=text_max_len, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    emb = encode_text(text_model, input_ids, attn_mask)
    return emb, attn_mask


# =============================================================================
# Model construction
# =============================================================================
def build_vq(args_dict: dict, device: torch.device) -> torch.nn.Module:
    vq = VQ_models[_arg(args_dict, "vq_model", "VQ-16")](
        codebook_size=_arg(args_dict, "codebook_size", 16384),
        codebook_embed_dim=_arg(args_dict, "codebook_embed_dim", 8),
    )
    return vq.to(device).eval()


def build_ar_t2i(
    args_dict: dict, latent_size: int, caption_dim: int, device: torch.device
) -> torch.nn.Module:
    """Reconstruct the dual-stream t2i AR exactly as ``train_ar_t2i.py`` does."""
    block_size = latent_size ** 2
    ar = LlamaGen_models[_arg(args_dict, "ar_model", "LlamaGen-B")](
        block_size=block_size,
        vocab_size=_arg(args_dict, "codebook_size", 16384),
        num_classes=_arg(args_dict, "num_classes", 1000),
        cls_token_num=_arg(args_dict, "cls_token_num", 300),
        caption_dim=caption_dim,
        text_attn=_arg(args_dict, "text_attn", "causal"),
        resid_dropout_p=_arg(args_dict, "dropout_p", 0.1),
        ffn_dropout_p=_arg(args_dict, "dropout_p", 0.1),
        token_dropout_p=_arg(args_dict, "token_dropout_p", 0.1),
        drop_path_rate=_arg(args_dict, "drop_path_rate", 0.0),
        use_checkpoint=False,
    )
    return ar.to(device).eval()


def load_ar_into(ar: torch.nn.Module, ckpt: dict, *, use_ema: bool, log_fn) -> None:
    if use_ema and "ema" in ckpt:
        sd, source = ckpt["ema"], "ema"
    elif "ar" in ckpt:
        sd, source = ckpt["ar"], "ar (live)"
    elif "ema" in ckpt:
        sd, source = ckpt["ema"], "ema (no live in ckpt)"
    else:
        raise KeyError("ckpt has none of {'ema', 'ar'}; cannot load AR weights.")
    missing, unexpected = ar.load_state_dict(_strip_compile_prefix(sd), strict=False)
    log_fn(f"[AR ] loaded '{source}'  missing={len(missing)}  unexpected={len(unexpected)}")


def load_vq_into(
    vq: torch.nn.Module, ckpt: dict, *, ckpt_path: Path, use_ema: bool,
    explicit_vq_ckpt: Path | None, log_fn,
) -> None:
    if explicit_vq_ckpt is not None:
        sd = load_pretrained_tokenizer_state_dict(str(explicit_vq_ckpt), use_ema=use_ema)
        source = f"--vq-ckpt-path {explicit_vq_ckpt} (use_ema={use_ema})"
    elif use_ema and "vq_ema" in ckpt:
        sd, source = ckpt["vq_ema"], "ckpt['vq_ema']"
    elif "vq" in ckpt:
        sd, source = ckpt["vq"], "ckpt['vq']"
    elif "vq_ema" in ckpt:
        sd, source = ckpt["vq_ema"], "ckpt['vq_ema'] (no live vq in ckpt)"
    else:
        saved_vq_ckpt = _arg(ckpt["args"], "vq_ckpt")
        if not saved_vq_ckpt:
            raise KeyError(
                f"ckpt {ckpt_path} has no 'vq'/'vq_ema' and ck['args']['vq_ckpt'] "
                f"is missing. Re-run with --vq-ckpt-path."
            )
        sd = load_pretrained_tokenizer_state_dict(str(saved_vq_ckpt), use_ema=use_ema)
        source = f"ck['args']['vq_ckpt'] = {saved_vq_ckpt} (use_ema={use_ema})"
    missing, unexpected = vq.load_state_dict(_strip_compile_prefix(sd), strict=False)
    log_fn(f"[VQ ] loaded {source}  missing={len(missing)}  unexpected={len(unexpected)}")


# =============================================================================
# Output layout helpers (mirror inference.py)
# =============================================================================
def _sampling_tag(args: argparse.Namespace) -> str:
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
    if override:
        return override
    if ckpt_path.parent.name == "checkpoints":
        return ckpt_path.parent.parent.name
    return ckpt_path.parent.name


def _step_tag(ckpt_path: Path) -> str:
    return f"step{ckpt_path.stem}"


def _make_log_fn(accelerator: Accelerator):
    def log(*msgs):
        if accelerator.is_main_process:
            print("[infer-t2i]", *msgs, flush=True)
    return log


def read_prompts(jsonl_path: Path, num: int) -> list[dict]:
    """Read the prompt JSONL; optionally truncate to the first ``num`` entries."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if num and num > 0:
        records = records[:num]
    return records


# =============================================================================
# Sampling loop
# =============================================================================
@torch.no_grad()
def sample_and_save(
    accelerator: Accelerator,
    ar: torch.nn.Module,
    vq: torch.nn.Module,
    tokenizer,
    text_model,
    uncond_emb_1: torch.Tensor,
    uncond_mask_1: torch.Tensor,
    prompts: list[dict],
    *,
    args: argparse.Namespace,
    args_dict: dict,
    sample_dir: Path,
    text_max_len: int,
    log_fn,
) -> int:
    """Distributed prompt-driven sampling -> one PNG per prompt.

    Returns the total number of prompts (== number of PNGs expected).
    """
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    image_size = _arg(args_dict, "image_size", 256)
    downsample = _arg(args_dict, "downsample_ratio", 16)
    latent_size = image_size // downsample
    codebook_embed_dim = _arg(args_dict, "codebook_embed_dim", 8)

    total = len(prompts)
    log_fn(
        f"prompts={total} world_size={world_size} "
        f"per_proc_batch_size={args.per_proc_batch_size}"
    )
    log_fn(
        f"sampling: cfg_scale={args.cfg_scale} cfg_interval={args.cfg_interval} "
        f"temperature={args.temperature} top_k={args.top_k} top_p={args.top_p} "
        f"seed={args.seed} image_size={image_size} latent_size={latent_size}"
    )

    if accelerator.is_main_process:
        sample_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    existing = {p.stem for p in sample_dir.glob("*.png")}
    if accelerator.is_main_process and existing:
        log_fn(f"resume: found {len(existing)} PNGs in {sample_dir}, will skip them.")

    # Each rank owns a disjoint, contiguous-stride slice of the prompt list.
    my_indices = list(range(rank, total, world_size))
    # Per-rank determinism for the AR multinomial sampling.
    torch.manual_seed(int(args.seed) * world_size + rank)

    ar_dtype = next(ar.parameters()).dtype
    n = int(args.per_proc_batch_size)
    n_batches = math.ceil(len(my_indices) / n) if my_indices else 0

    pbar = tqdm(range(n_batches), disable=not accelerator.is_main_process,
                desc="sample", unit="batch")
    t0 = time.time()
    for b in pbar:
        batch_idx = my_indices[b * n:(b + 1) * n]
        # Skip a batch entirely only if every PNG already exists.
        todo = [i for i in batch_idx if f"{i:06d}" not in existing]
        if not todo:
            continue

        captions = [prompts[i]["caption"] for i in todo]
        cond_emb, cond_mask = encode_prompts(
            tokenizer, text_model, captions, text_max_len, device,
        )
        cond_emb = cond_emb.to(ar_dtype)
        bsz = cond_emb.shape[0]

        if args.cfg_scale > 1.0:
            uncond_emb = uncond_emb_1.to(ar_dtype).expand(bsz, -1, -1)
            uncond_mask = uncond_mask_1.expand(bsz, -1)
        else:
            uncond_emb, uncond_mask = None, None

        index_sample = generate_t2i(
            ar, cond_emb, uncond_emb, latent_size ** 2,
            cond_mask=cond_mask, uncond_mask=uncond_mask,
            cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            sample_logits=True,
        )
        qzshape = [bsz, int(codebook_embed_dim), latent_size, latent_size]
        samples = vq.decode_code(index_sample, qzshape)  # [-1, 1]
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
        for j, gi in enumerate(todo):
            Image.fromarray(np.ascontiguousarray(samples[j])).save(
                str(sample_dir / f"{gi:06d}.png")
            )

    accelerator.wait_for_everyone()
    log_fn(f"sampling done in {time.time() - t0:.1f}s -> {sample_dir}")
    return total


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ---- checkpoint ----------------------------------------------------------
    p.add_argument("--ckpt-path", type=Path, required=True,
                   help="Stage-2 t2i checkpoint .pt from train_ar_t2i.py.")
    p.add_argument("--vq-ckpt-path", type=Path, default=None,
                   help="Optional override for the VQ weights (needed when the "
                        "t2i ckpt's saved ck['args']['vq_ckpt'] is unreachable).")
    p.add_argument("--use-ar-ema", default=True, action=argparse.BooleanOptionalAction,
                   help="Use AR EMA weights (default). --no-use-ar-ema for live AR.")
    p.add_argument("--use-vq-ema", default=True, action=argparse.BooleanOptionalAction,
                   help="Use VQ EMA weights (default). --no-use-vq-ema for live VQ.")

    # ---- prompts -------------------------------------------------------------
    p.add_argument("--prompts-jsonl", type=Path, required=True,
                   help="JSONL of {key, caption, caption_type} (build with "
                        "tools/build_gpic_prompts.py for a matched GPIC eval).")
    p.add_argument("--num", type=int, default=0,
                   help="Use only the first N prompts (0 = all, default).")

    # ---- text encoder (defaults read from ckpt args unless overridden) -------
    p.add_argument("--text-encoder", type=str, default="",
                   help="Frozen text encoder path or HF id (e.g. Qwen/Qwen3-1.7B). "
                        "Empty => use ckpt args.text_encoder (REQUIRED for "
                        "published weights whose args were stripped on upload).")
    p.add_argument("--text-max-len", type=int, default=0,
                   help="Caption token length. 0 => ckpt args.text_max_len, else 300.")

    # ---- model architecture (override ckpt args; REQUIRED for published HF
    #      weights, whose args snapshot is stripped on upload) -----------------
    p.add_argument("--ar-model", type=str, default=None,
                   help="AR family, e.g. LlamaGen-XL / LlamaGen-1B. Required for HF weights.")
    p.add_argument("--image-size", type=int, default=None,
                   help="Generation resolution the AR was trained at (256/512).")
    p.add_argument("--vq-model", type=str, default=None,
                   help="Tokenizer family (default VQ-16).")
    p.add_argument("--codebook-size", type=int, default=None,
                   help="Codebook size (default 16384).")
    p.add_argument("--codebook-embed-dim", type=int, default=None,
                   help="Codebook embedding dim (VQ=8, LFQ=14, IBQ=256).")
    p.add_argument("--downsample-ratio", type=int, default=None,
                   help="VQ spatial downsample ratio (default 16).")
    p.add_argument("--cls-token-num", type=int, default=None,
                   help="Text-prefix length; must equal --text-max-len (default 300).")
    p.add_argument("--text-attn", type=str, default=None, choices=["causal", "prefix"],
                   help="Text-prefix attention topology (default causal).")

    # ---- sampling ------------------------------------------------------------
    p.add_argument("--per-proc-batch-size", type=int, default=32)
    p.add_argument("--cfg-scale", type=float, default=1.0,
                   help="CFG scale (1.0 = no CFG, the GPIC default).")
    p.add_argument("--cfg-interval", type=float, default=-1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)

    # ---- output --------------------------------------------------------------
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Parent dir; writes <output-dir>/<exp>/<step>/<tag>/"
                        "{samples/, samples.npz}.")
    p.add_argument("--exp-name", type=str, default="")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--image-size-eval", type=int, default=None,
                   help="Resize generated images before saving (default: trained size).")
    p.add_argument("--keep-pngs", default=True, action=argparse.BooleanOptionalAction,
                   help="Keep per-image PNGs after the npz is packed (default: keep).")
    p.add_argument("--pack-npz", default=True, action=argparse.BooleanOptionalAction,
                   help="Pack the PNGs into samples.npz (arr_0 NHWC uint8). "
                        "--no-pack-npz to only keep the PNG directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    accelerator = Accelerator()
    log_fn = _make_log_fn(accelerator)
    device = accelerator.device

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    log_fn(f"world_size={accelerator.num_processes} device={device}")
    log_fn(f"loading ckpt: {args.ckpt_path}")

    ck = load_ckpt(args.ckpt_path, map_location="cpu")
    had_saved_args = bool(ck["args"])
    args_dict = _overlay_cli_overrides(ck["args"], args)
    if not had_saved_args:
        log_fn("ckpt has no saved 'args' (published weights); using CLI flags "
               "+ canonical defaults for the architecture.")
    if args.image_size_eval is None:
        args.image_size_eval = _arg(args_dict, "image_size", 256)

    text_encoder_path = args.text_encoder or _arg(args_dict, "text_encoder")
    text_max_len = args.text_max_len or _arg(args_dict, "text_max_len", 300)
    if not text_encoder_path:
        raise ValueError(
            "no text encoder path: pass --text-encoder (ckpt args has none)."
        )
    log_fn(
        f"ckpt args: ar={_arg(args_dict, 'ar_model')} vq={_arg(args_dict, 'vq_model')} "
        f"image_size={_arg(args_dict, 'image_size')} cls_token_num={_arg(args_dict, 'cls_token_num')} "
        f"text_attn={_arg(args_dict, 'text_attn', 'causal')} "
        f"text_encoder={text_encoder_path} text_max_len={text_max_len} "
        f"steps={ck.get('steps', '?')}"
    )

    image_size = _arg(args_dict, "image_size", 256)
    downsample = _arg(args_dict, "downsample_ratio", 16)
    latent_size = image_size // downsample

    # Text encoder dtype follows the trainer's mixed-precision setting.
    mp = _arg(args_dict, "mixed_precision", "bf16")
    text_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mp, torch.float32)
    tokenizer, text_model, text_hidden = load_text_encoder(
        text_encoder_path, device, text_dtype,
    )
    uncond_emb_1, uncond_mask_1 = encode_prompts(
        tokenizer, text_model, [""], text_max_len, device,
    )
    log_fn(f"text encoder hidden={text_hidden} | "
           f"uncond valid_tokens={int(uncond_mask_1.sum().item())}")

    vq = build_vq(args_dict, device)
    ar = build_ar_t2i(args_dict, latent_size, text_hidden, device)

    load_vq_into(
        vq, ck, ckpt_path=args.ckpt_path, use_ema=args.use_vq_ema,
        explicit_vq_ckpt=args.vq_ckpt_path, log_fn=log_fn,
    )
    load_ar_into(ar, ck, use_ema=args.use_ar_ema, log_fn=log_fn)
    del ck

    prompts = read_prompts(args.prompts_jsonl, args.num)
    log_fn(f"loaded {len(prompts)} prompts from {args.prompts_jsonl}")

    exp_name = _resolve_exp_name(args.ckpt_path, args.exp_name)
    step_tag = _step_tag(args.ckpt_path)
    sampling_tag = _sampling_tag(args)
    run_dir = args.output_dir / exp_name / step_tag / sampling_tag
    sample_dir = run_dir / "samples"
    npz_path = run_dir / "samples.npz"
    log_fn(f"run_dir={run_dir}")

    if accelerator.is_main_process and npz_path.exists():
        log_fn(f"FOUND existing {npz_path}; nothing to do. Delete it (or pass a "
               f"different --tag) to re-sample.")
    accelerator.wait_for_everyone()
    if npz_path.exists():
        return

    total = sample_and_save(
        accelerator, ar, vq, tokenizer, text_model, uncond_emb_1, uncond_mask_1,
        prompts, args=args, args_dict=args_dict, sample_dir=sample_dir,
        text_max_len=text_max_len, log_fn=log_fn,
    )

    if accelerator.is_main_process:
        if args.pack_npz:
            log_fn(f"packing NPZ: {npz_path}")
            npz_made = create_npz_from_sample_folder(str(sample_dir), int(total))
            if Path(npz_made) != npz_path:
                shutil.move(str(npz_made), str(npz_path))
            log_fn(f"NPZ ready: {npz_path}")

        if not args.keep_pngs:
            try:
                shutil.rmtree(sample_dir)
                log_fn(f"deleted PNG folder {sample_dir}")
            except OSError as exc:
                log_fn(f"WARN failed to clean {sample_dir}: {exc}")

        ref_hint = "/path/to/gpic/reference_stats/test_stats.npz"
        eval_target = npz_path if args.pack_npz else sample_dir
        log_fn(
            "\n  --- next step: GPIC FD-DINOv2 eval (in-repo, proxy set for DINOv2 dl) ---\n"
            f"    python src/eval/gpic/gpic_eval_dino.py \\\n"
            f"        {eval_target} \\\n"
            f"        {ref_hint} \\\n"
            f"        --metrics fd,prdc,mmd\n"
        )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
