"""Shared text-to-image generation helpers for the ``src`` eval benchmarks.

The GenEval / DPG-Bench step-1 scripts only differ in *how the prompts are laid
out on disk*; the actual model loading and sampling are identical and are the
same code path used by ``src/inference_t2i.py`` (the GPIC eval). To avoid
duplicating that logic, both step-1 scripts build a model "bundle" here and call
:func:`generate_images` to turn a list of captions into PIL images.

This mirrors ``inference_t2i.py``'s pipeline exactly:
    1. Load the Stage-2 t2i checkpoint, the frozen Qwen text encoder, the dual
       stream AR (``models.llamagen_t2i``) and the frozen VQ tokenizer.
    2. For a batch of captions, encode text -> ``generate_t2i`` (AR sampling with
       optional CFG) -> ``vq.decode_code`` -> uint8 RGB images.

Everything is imported from ``src.inference_t2i`` so there is a single
source of truth for the checkpoint plumbing.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from models.generate_t2i import generate_t2i  # noqa: E402
from src.inference_t2i import (  # noqa: E402
    _arg,
    _overlay_cli_overrides,
    _resolve_exp_name,
    _sampling_tag,
    _step_tag,
    build_ar_t2i,
    build_vq,
    encode_prompts,
    load_ar_into,
    load_ckpt,
    load_text_encoder,
    load_vq_into,
)

# Re-export the layout helpers so the step-1 scripts can build run dirs that
# match the GPIC eval ("<output>/<exp>/<step>/<tag>/...").
__all__ = [
    "ModelBundle",
    "add_model_args",
    "load_model_bundle",
    "generate_images",
    "resolve_run_dir",
]


@dataclass
class ModelBundle:
    accelerator: Accelerator
    ar: torch.nn.Module
    vq: torch.nn.Module
    tokenizer: object
    text_model: torch.nn.Module
    uncond_emb_1: torch.Tensor
    uncond_mask_1: torch.Tensor
    args_dict: dict
    text_max_len: int
    image_size: int
    latent_size: int
    codebook_embed_dim: int
    ckpt_path: Path
    exp_name: str = field(default="")

    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    @property
    def is_main(self) -> bool:
        return self.accelerator.is_main_process


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add the checkpoint / text-encoder / sampling flags shared by every bench.

    These deliberately mirror ``inference_t2i.py`` so the eval scripts behave the
    same way as the GPIC eval (same EMA defaults, same CFG / sampling knobs).
    """
    g = parser.add_argument_group("src model")
    g.add_argument("--ckpt-path", type=Path, required=True,
                   help="Stage-2 t2i checkpoint .pt from train_ar_t2i.py.")
    g.add_argument("--vq-ckpt-path", type=Path, default=None,
                   help="Optional override for the VQ tokenizer weights.")
    g.add_argument("--use-ar-ema", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="Use AR EMA weights (default).")
    g.add_argument("--use-vq-ema", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="Use VQ EMA weights (default).")
    g.add_argument("--text-encoder", type=str, default="",
                   help="Frozen text encoder path or HF id (e.g. Qwen/Qwen3-1.7B). "
                        "Empty => use ckpt args (REQUIRED for published weights "
                        "whose args were stripped on upload).")
    g.add_argument("--text-max-len", type=int, default=0,
                   help="Caption token length. 0 => ckpt args.text_max_len, else 300.")
    # Architecture overrides (REQUIRED for published HF weights whose args
    # snapshot is stripped on upload). None => ckpt args, then canonical default.
    g.add_argument("--ar-model", type=str, default=None,
                   help="AR family, e.g. LlamaGen-XL / LlamaGen-1B. Required for HF weights.")
    g.add_argument("--image-size", type=int, default=None,
                   help="Generation resolution the AR was trained at (256/512).")
    g.add_argument("--vq-model", type=str, default=None, help="Tokenizer family (default VQ-16).")
    g.add_argument("--codebook-size", type=int, default=None, help="Codebook size (default 16384).")
    g.add_argument("--codebook-embed-dim", type=int, default=None,
                   help="Codebook embedding dim (VQ=8, LFQ=14, IBQ=256).")
    g.add_argument("--downsample-ratio", type=int, default=None,
                   help="VQ spatial downsample ratio (default 16).")
    g.add_argument("--cls-token-num", type=int, default=None,
                   help="Text-prefix length; must equal --text-max-len (default 300).")
    g.add_argument("--text-attn", type=str, default=None, choices=["causal", "prefix"],
                   help="Text-prefix attention topology (default causal).")

    s = parser.add_argument_group("sampling")
    s.add_argument("--cfg-scale", type=float, default=1.0,
                   help="CFG scale (1.0 = no CFG).")
    s.add_argument("--cfg-interval", type=float, default=-1)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--top-k", type=int, default=0)
    s.add_argument("--top-p", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--tag", type=str, default="")
    s.add_argument("--exp-name", type=str, default="")


def load_model_bundle(args: argparse.Namespace) -> ModelBundle:
    """Construct the full src t2i model stack on the current accelerate rank."""
    accelerator = Accelerator()
    device = accelerator.device

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    def log(*m):
        if accelerator.is_main_process:
            print("[eval-infer]", *m, flush=True)

    log(f"world_size={accelerator.num_processes} device={device}")
    log(f"loading ckpt: {args.ckpt_path}")

    ck = load_ckpt(args.ckpt_path, map_location="cpu")
    had_saved_args = bool(ck["args"])
    args_dict = _overlay_cli_overrides(ck["args"], args)
    if not had_saved_args:
        log("ckpt has no saved 'args' (published weights); using CLI flags "
            "+ canonical defaults for the architecture.")

    text_encoder_path = args.text_encoder or _arg(args_dict, "text_encoder")
    text_max_len = args.text_max_len or _arg(args_dict, "text_max_len", 300)
    if not text_encoder_path:
        raise ValueError("no text encoder path: pass --text-encoder (ckpt has none).")

    image_size = _arg(args_dict, "image_size", 256)
    downsample = _arg(args_dict, "downsample_ratio", 16)
    latent_size = image_size // downsample
    codebook_embed_dim = _arg(args_dict, "codebook_embed_dim", 8)

    mp = _arg(args_dict, "mixed_precision", "bf16")
    text_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mp, torch.float32)
    tokenizer, text_model, text_hidden = load_text_encoder(
        text_encoder_path, device, text_dtype,
    )
    uncond_emb_1, uncond_mask_1 = encode_prompts(
        tokenizer, text_model, [""], text_max_len, device,
    )
    log(f"text encoder hidden={text_hidden} image_size={image_size} "
        f"latent_size={latent_size} text_max_len={text_max_len}")

    vq = build_vq(args_dict, device)
    ar = build_ar_t2i(args_dict, latent_size, text_hidden, device)
    load_vq_into(vq, ck, ckpt_path=args.ckpt_path, use_ema=args.use_vq_ema,
                 explicit_vq_ckpt=args.vq_ckpt_path, log_fn=log)
    load_ar_into(ar, ck, use_ema=args.use_ar_ema, log_fn=log)
    del ck

    return ModelBundle(
        accelerator=accelerator, ar=ar, vq=vq, tokenizer=tokenizer,
        text_model=text_model, uncond_emb_1=uncond_emb_1,
        uncond_mask_1=uncond_mask_1, args_dict=args_dict,
        text_max_len=text_max_len, image_size=image_size,
        latent_size=latent_size, codebook_embed_dim=codebook_embed_dim,
        ckpt_path=args.ckpt_path,
        exp_name=_resolve_exp_name(args.ckpt_path, args.exp_name),
    )


@torch.no_grad()
def generate_images(
    bundle: ModelBundle,
    captions: list[str],
    args: argparse.Namespace,
    *,
    resize: int | None = None,
) -> list[Image.Image]:
    """Sample one image per caption. Returns a list of PIL RGB images.

    Stochastic AR sampling means duplicate captions in ``captions`` yield
    different images (this is how GenEval's ``n_samples`` per prompt and
    DPG-Bench's 4-image grid get their variety).
    """
    device = bundle.device
    ar_dtype = next(bundle.ar.parameters()).dtype
    latent_size = bundle.latent_size

    cond_emb, cond_mask = encode_prompts(
        bundle.tokenizer, bundle.text_model, captions,
        bundle.text_max_len, device,
    )
    cond_emb = cond_emb.to(ar_dtype)
    bsz = cond_emb.shape[0]

    if args.cfg_scale > 1.0:
        uncond_emb = bundle.uncond_emb_1.to(ar_dtype).expand(bsz, -1, -1)
        uncond_mask = bundle.uncond_mask_1.expand(bsz, -1)
    else:
        uncond_emb, uncond_mask = None, None

    index_sample = generate_t2i(
        bundle.ar, cond_emb, uncond_emb, latent_size ** 2,
        cond_mask=cond_mask, uncond_mask=uncond_mask,
        cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        sample_logits=True,
    )
    qzshape = [bsz, int(bundle.codebook_embed_dim), latent_size, latent_size]
    samples = bundle.vq.decode_code(index_sample, qzshape)  # [-1, 1]
    samples = (
        torch.clamp(127.5 * samples + 128.0, 0, 255)
        .permute(0, 2, 3, 1)
        .to("cpu", dtype=torch.uint8)
        .numpy()
    )
    images = []
    for j in range(bsz):
        img = Image.fromarray(np.ascontiguousarray(samples[j]))
        if resize is not None and resize != img.size[0]:
            img = img.resize((resize, resize), Image.BICUBIC)
        images.append(img)
    return images


def resolve_run_dir(bundle: ModelBundle, args: argparse.Namespace,
                    output_dir: Path) -> Path:
    """Build ``<output>/<exp>/<step>/<sampling-tag>`` like the GPIC eval does."""
    return (Path(output_dir) / bundle.exp_name
            / _step_tag(bundle.ckpt_path) / _sampling_tag(args))
