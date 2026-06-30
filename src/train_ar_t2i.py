"""Stage 2 of the src pipeline: train AR with the frozen end-to-end VQ.

Differences vs. Stage 1:

* The native VQ model (`models/vq_model.py`) is loaded from a Stage-1
  checkpoint and **frozen**. We do not need the reconstruction / GAN /
  perceptual losses.
* AR receives **hard token indices** (``argmin d``), matching the
  inference distribution.
* REPA loss is still computed on the AR's intermediate layer; it only
  updates the AR weights + the projector(s).

Online encoding: every batch reads raw images from the H5 dataset (same
as Stage 1) and runs them through the frozen VQ to get token indices on
the fly. No re-caching needed when the Stage-1 VQ changes.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path

# CRITICAL: the HF (Rust) tokenizer is used inside forked DataLoader workers
# (WebDatasetT2IStream._tokenize). With parallelism enabled, using a tokenizer
# in the parent and then again in a forked child DEADLOCKS the worker (no NCCL
# op in flight -> no watchdog timeout -> a silent, seed/timing-dependent hang).
# Force it off here, before transformers is imported, regardless of the shell.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, ProjectConfiguration, set_seed
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from models import Tokenizers as VQ_models  # noqa: E402
# Unified registry that holds the VQ (models/vq_model.py), LFQ
# (models/lfq_model.py) and IBQ (models/ibq_model.py) families. Aliased
# back to ``VQ_models`` so the existing ``VQ_models[args.vq_model]`` lookup
# is unchanged -- pass ``--vq-model=LFQ-16`` / ``IBQ-16`` (or ``VQ-16`` /
# ``VQ-8``) at the CLI to switch.
#
# NOTE: this Stage-2 t2i trainer uses the dual-stream (MMDiT-style joint
# self-attention) LlamaGen variant in ``models/llamagen_t2i.py`` -- the text
# stream has its own qkv/o + attention_norm and the FFN is shared with the
# image stream. A class-conditional (c2i) checkpoint loads into the image
# stream with ``strict=False``; the text-stream params and CaptionEmbedder are
# newly initialised (optionally warm-copied from the image stream).
from models.llamagen_t2i import LlamaGen_models  # noqa: E402
from models.generate_t2i import generate_t2i  # noqa: E402  - t2i AR sampling

from transformers import AutoModel, AutoTokenizer  # noqa: E402

from src.blip3o_dataset import prepare_blip3o_t2i_dataloader  # noqa: E402
from src.gpic_dataset import prepare_gpic_t2i_dataloader  # noqa: E402
from src.coco_fid import CocoFIDEvaluator  # noqa: E402  - t2i COCO FID
from src.utils import (  # noqa: E402
    count_trainable_params,
    extract_at_layer,
    extract_repa_target,
    load_encoders,
    load_pretrained_tokenizer_state_dict,
    num_encoder_layers,
    preprocess_imgs_for_codec,
    preprocess_raw_image,
    run_distributed_fid_eval,
    spatial_norm,
)


logger = get_logger(__name__)


# =============================================================================
# Helpers
# =============================================================================
def array2grid(x, nrow=None):
    if nrow is None:
        nrow = round(math.sqrt(x.size(0)))
    g = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    g = g.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return g


def array2grid_pairs(inputs, recons, pairs_per_row: int = 4):
    """Lay out (input, recon) pairs side-by-side. See src/train_gear.py."""
    assert inputs.shape == recons.shape
    pairs = torch.stack([inputs, recons], dim=1).reshape(-1, *inputs.shape[1:])
    return array2grid(pairs, nrow=2 * pairs_per_row)


def build_repa_projector(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class ProjectorBank(nn.Module):
    """See :class:`src.train_gear.ProjectorBank` -- same wrapper so DDP
    can wrap the bank as a single Module (a DDP-wrapped ``nn.ModuleList`` is
    not iterable, which breaks the per-encoder loop)."""

    def __init__(self, in_dim: int, hidden: int, out_dims):
        super().__init__()
        self.projs = nn.ModuleList(
            [build_repa_projector(in_dim, hidden, d) for d in out_dims]
        )

    def __len__(self) -> int:
        return len(self.projs)

    def forward(self, patch_hidden: torch.Tensor):
        return [proj(patch_hidden) for proj in self.projs]


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
    )
    return logging.getLogger(__name__)


def repa_align_loss(zs_tilde_list, zs_list):
    proj = torch.zeros((), device=zs_tilde_list[0].device)
    bsz = zs_list[0].shape[0]
    for z, z_tilde in zip(zs_list, zs_tilde_list):
        for z_j, z_tilde_j in zip(z, z_tilde):
            z_tilde_j = F.normalize(z_tilde_j, dim=-1)
            z_j = F.normalize(z_j, dim=-1)
            proj = proj + (-(z_j * z_tilde_j).sum(dim=-1)).mean()
    return proj / (len(zs_list) * bsz)


def project_repa(hidden_at_tap, cls_token_num, projectors):
    patch_hidden = hidden_at_tap[:, cls_token_num - 1:]
    return projectors(patch_hidden)


# Fixed prompts used for the periodic wandb sample grid (t2i sanity check).
SAMPLE_PROMPTS = [
    "A photo of an astronaut riding a horse on the moon.",
    "A bowl of fresh strawberries on a wooden table, soft morning light.",
    "A red sports car parked in front of a yellow brick wall.",
    "A cute corgi puppy sitting in a field of flowers.",
]


def load_text_encoder(path, device, dtype):
    """Load a frozen Qwen text encoder + tokenizer.

    Supports both a plain text LLM (e.g. Qwen3-1.7B -> ``AutoModel`` returns a
    ``Qwen3Model`` directly) and a VLM (e.g. Qwen3.5 -> ``AutoModel`` returns a
    model with ``.language_model`` text tower + ``.visual`` vision tower, of
    which we keep only the text tower). Returns ``(tokenizer, text_model,
    hidden_size)``.
    """
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    # ``torch_dtype`` is the canonical arg across transformers 4.x/5.x (the
    # ``dtype`` alias only exists on newer versions).
    full = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=dtype)
    if hasattr(full, "language_model"):
        text_model = full.language_model
        if hasattr(full, "visual"):
            del full.visual
    else:
        text_model = full
    text_model = text_model.to(device).eval()
    for p in text_model.parameters():
        p.requires_grad = False
    hidden_size = text_model.config.hidden_size
    return tokenizer, text_model, hidden_size


@torch.no_grad()
def encode_text(text_model, input_ids, attn_mask):
    """Frozen Qwen forward -> last_hidden_state (B, L, hidden)."""
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


@torch.no_grad()
def ar_generate_pil(
    ar_model, vq, tokenizer, text_model, uncond_emb_1, uncond_mask_1, caption,
    *, latent_size, codebook_embed_dim, text_max_len, device,
    cfg_scale, temperature, top_k, top_p,
):
    """Generate a single PIL image from one caption (for COCO FID).

    Runs WITHOUT autocast: ``setup_caches`` allocates the KV cache in the AR
    param dtype (fp32 under accelerate mixed precision), so we cast the Qwen
    text embeddings (bf16) to the AR dtype before ``generate_t2i``.
    """
    from PIL import Image
    cond_emb, cond_mask = encode_prompts(tokenizer, text_model, [caption], text_max_len, device)
    ar_dtype = next(ar_model.parameters()).dtype
    cond_emb = cond_emb.to(ar_dtype)
    uncond_emb = uncond_emb_1.to(ar_dtype)
    index_sample = generate_t2i(
        ar_model, cond_emb, uncond_emb, latent_size ** 2,
        cond_mask=cond_mask, uncond_mask=uncond_mask_1,
        cfg_scale=cfg_scale, cfg_interval=-1,
        temperature=temperature, top_k=top_k, top_p=top_p, sample_logits=True,
    )
    qzshape = [1, codebook_embed_dim, latent_size, latent_size]
    img = vq.decode_code(index_sample, qzshape)         # [-1, 1], (1, 3, H, W)
    img01 = (img.clamp(-1, 1) + 1) / 2.0
    arr = (img01[0] * 255).round().clamp(0, 255).to("cpu", torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


# =============================================================================
# Main
# =============================================================================
def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    # 10-min NCCL collective timeout: turns a silent hang into a real exception
    # with a Python traceback (default 30 min is too long for debugging). With
    # run.sh's TORCH_NCCL_BLOCKING_WAIT=1 this surfaces any rank skew quickly.
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=10))
    # IMPORTANT: ``rng_types=[]`` disables accelerate's RNG ``broadcast(..., src=0)``
    # (issued inside its DataLoader ``__iter__`` / per-step RNG sync). On many
    # ranks that broadcast is a classic collective-ordering deadlock source:
    # if one rank issues it a step ahead of others (from accumulated NCCL skew,
    # which is worse with uneven streaming data), it blocks while the others are
    # still in the previous step's all-reduce. We drive shuffling ourselves in
    # the dataset (seeded per rank/worker), so we don't need accelerate's RNG sync.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
        kwargs_handlers=[pg_kwargs],
        rng_types=[],
    )

    save_dir = os.path.join(args.output_dir, args.exp_name)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
        ckpt_dir = f"{save_dir}/checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        log = create_logger(save_dir)
        log.info(f"Experiment directory created at {save_dir}")

    device = accelerator.device
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # =========================================================================
    # 1. Build & load the (frozen) VQ from a Stage-1 checkpoint
    # =========================================================================
    vq = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    )
    # Same helper handles native ``{"vq": ..., "vq_ema": ...}`` (our
    # Stage-0/1 ckpts), legacy ``{"model": ...}`` (the public
    # ``vq_ds16_c2i.pt``) and TencentARC MAGVIT2 lightning ckpts (LFQ
    # pretrain). A single flag works for both VQ-16 and LFQ-16 runs.
    #
    # ``--vq-use-ema`` toggles which tensor of *our* Stage-0/1 ckpts is
    # loaded: the live params (``vq``, default) or the EMA shadow
    # (``vq_ema``). EMA is what Stage 1 itself reports under
    # ``vq_val_ema/*`` and is the recommended choice for the frozen
    # tokenizer in Stage 2 / inference; the live copy is kept around
    # for ablations and continued training. Has no effect on legacy /
    # MAGVIT2 ckpts (those expose only one tensor source).
    sd = load_pretrained_tokenizer_state_dict(
        args.vq_ckpt, use_ema=args.vq_use_ema,
    )
    missing, unexpected = vq.load_state_dict(sd, strict=False)
    if accelerator.is_main_process:
        log.info(
            f"Loaded VQ from {args.vq_ckpt} "
            f"(use_ema={args.vq_use_ema}) | "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    vq = vq.to(device).eval()
    for p in vq.parameters():
        p.requires_grad = False

    # =========================================================================
    # 1b. Frozen text encoder (Qwen3.5) + precomputed uncond (empty-string) emb
    # =========================================================================
    if args.mixed_precision == "fp16":
        text_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        text_dtype = torch.bfloat16
    else:
        text_dtype = torch.float32
    tokenizer, text_model, text_hidden = load_text_encoder(
        args.text_encoder, device, text_dtype,
    )
    # The unconditional branch (for CFG) is the empty-string embedding -- fixed,
    # not learnable. Precompute it once and reuse for caption dropout + sampling.
    uncond_emb_1, uncond_mask_1 = encode_prompts(
        tokenizer, text_model, [""], args.text_max_len, device,
    )  # (1, T, H), (1, T)
    if accelerator.is_main_process:
        log.info(
            f"Loaded text encoder {args.text_encoder} | hidden={text_hidden} | "
            f"uncond(empty) valid_tokens={int(uncond_mask_1.sum().item())}"
        )

    # =========================================================================
    # 2. AR + projectors
    # =========================================================================
    assert args.image_size % args.downsample_ratio == 0
    assert args.text_max_len == args.cls_token_num, (
        f"--text-max-len ({args.text_max_len}) must equal --cls-token-num "
        f"({args.cls_token_num}): the padded caption length IS the AR text prefix."
    )
    latent_size = args.image_size // args.downsample_ratio
    block_size = latent_size ** 2

    if args.use_repa:
        encoders, encoder_types, _ = load_encoders(args.enc_type, device, args.image_size)
        z_dims = [enc.embed_dim for enc in encoders]
        if args.repa_encoder_layer != -1:
            for enc, etype in zip(encoders, encoder_types):
                n_blk = num_encoder_layers(enc, etype)
                if not (1 <= args.repa_encoder_layer <= n_blk):
                    raise ValueError(
                        f"--repa-encoder-layer={args.repa_encoder_layer} is out of "
                        f"range for {etype} (has {n_blk} blocks; valid: -1 or "
                        f"1..{n_blk})."
                    )
    else:
        encoders, encoder_types, z_dims = None, None, None
        if accelerator.is_main_process:
            log.info("REPA disabled (--no-use-repa): skipping encoder load; AR will be trained with CE only.")

    ar = LlamaGen_models[args.ar_model](
        block_size=block_size,
        vocab_size=args.codebook_size,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        caption_dim=text_hidden,
        text_attn=args.text_attn,
        resid_dropout_p=args.dropout_p,
        ffn_dropout_p=args.dropout_p,
        token_dropout_p=args.token_dropout_p,
        drop_path_rate=args.drop_path_rate,
        use_checkpoint=args.use_checkpoint,
    ).to(device)
    if args.use_repa and not (1 <= args.encoder_depth <= len(ar.layers)):
        raise ValueError(f"--encoder-depth={args.encoder_depth} out of range [1, {len(ar.layers)}].")

    if args.use_repa:
        projectors = ProjectorBank(
            in_dim=ar.config.dim, hidden=args.projector_dim, out_dims=z_dims,
        ).to(device)
    else:
        projectors = None

    if args.ar_init_ckpt:
        ar_ck = torch.load(args.ar_init_ckpt, map_location="cpu")
        sd_ar = ar_ck.get("ema", ar_ck.get("ar", ar_ck))
        missing, unexpected = ar.load_state_dict(sd_ar, strict=False)
        if accelerator.is_main_process:
            log.info(f"Loaded AR init from {args.ar_init_ckpt} | missing={len(missing)}, unexpected={len(unexpected)}")
            log.info(f"  (expected missing: text-stream wqkv_text/wo_text/attention_norm_text + cls_embedding.cap_proj; "
                     f"unexpected: c2i cls_embedding.embedding_table)")
        # Warm-start the new text stream from the image stream so it begins
        # in the same regime as the pretrained c2i weights. Only when freshly
        # initialising from a c2i ckpt (not when resuming a t2i ckpt below).
        if args.text_stream_copy_init and args.resume_step <= 0:
            ar.init_text_stream_from_image()
            if accelerator.is_main_process:
                log.info("Warm-copied image-stream attention params -> text stream.")
        if projectors is not None and "projectors" in ar_ck:
            projectors.load_state_dict(ar_ck["projectors"])

    ema = copy.deepcopy(ar).to(device)
    for p in ema.parameters():
        p.requires_grad = False

    if accelerator.is_main_process:
        log.info(f"AR params       : {sum(p.numel() for p in ar.parameters()):,}")
        if projectors is not None:
            log.info(f"Projector params: {sum(p.numel() for p in projectors.parameters()):,}")
        log.info(f"Trainable AR    : {count_trainable_params(ar):,}")

    # =========================================================================
    # 3. Optim, data
    # =========================================================================
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    opt_params = list(ar.parameters())
    if projectors is not None:
        opt_params = opt_params + list(projectors.parameters())
    opt = torch.optim.AdamW(
        opt_params,
        lr=args.ar_lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    local_bs = int(args.batch_size // accelerator.num_processes)
    # Streaming webdataset (direct tarfile reader). The dataset already shards
    # tars across (rank x DataLoader worker) internally and device placement is
    # manual in the loop, so we deliberately do NOT pass this loader to
    # accelerator.prepare (that would re-dispatch / re-shard the IterableDataset).
    # Per-source interleave probabilities (e.g. dataset sizes). Streaming
    # datasets have unknown length so HF cannot infer these -- pass them
    # explicitly; they are normalised inside the loader. With probs proportional
    # to dataset sizes and stopping_strategy="all_exhausted", both sources
    # deplete together so ~1 epoch == each sample seen ~once.
    data_probs = None
    if args.data_probs:
        data_probs = [float(x) for x in args.data_probs.split(",") if x.strip()]
    _loader_common = dict(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=local_bs,
        image_size=args.image_size,
        text_max_len=args.text_max_len,
        num_workers=args.num_workers,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        seed=args.seed if args.seed is not None else 0,
        random_hflip=args.random_hflip,
        data_probs=data_probs,
    )
    if args.data_type == "gpic":
        # GPIC: {key}.json (caption inside) + {key}.jpg/.png. All caption
        # types (tag/short/medium/long) are kept.
        train_loader, _ = prepare_gpic_t2i_dataloader(**_loader_common)
    else:
        # BLIP3o: {key}.jpg + {key}.txt
        train_loader, _ = prepare_blip3o_t2i_dataloader(**_loader_common)
    if accelerator.is_main_process:
        log.info(f"Streaming {args.data_type} webdataset from {args.data_dir} "
                 f"(local_bs={local_bs}, image_size={args.image_size}, "
                 f"text_max_len={args.text_max_len}, data_probs={data_probs})")

    update_ema(ema, ar, decay=0.0)
    ar.eval(); ema.eval()

    global_step = 0
    if args.resume_step > 0:
        path = f"{save_dir}/checkpoints/{args.resume_step:07d}.pt"
        ck = torch.load(path, map_location="cpu")
        ar.load_state_dict(ck["ar"]); ema.load_state_dict(ck["ema"])
        if projectors is not None and "projectors" in ck:
            projectors.load_state_dict(ck["projectors"])
        opt.load_state_dict(ck["opt"]); global_step = ck["steps"]

    # ---- step budget --------------------------------------------------------
    # The streaming dataset cycles INFINITELY (each rank's loader never raises
    # StopIteration), avoiding the streaming+DDP deadlock where a rank that
    # exhausts its (uneven) shard first desyncs the per-step collectives. So we
    # stop purely on a step budget -- no per-batch collective. Setting
    # ``--samples-per-epoch`` (total images / epoch) expresses "N epochs":
    # steps_per_epoch = ceil(samples_per_epoch / global_batch_size).
    effective_max_steps = args.max_train_steps
    if args.samples_per_epoch and args.samples_per_epoch > 0:
        steps_per_epoch = math.ceil(args.samples_per_epoch / args.batch_size)
        effective_max_steps = min(args.max_train_steps, args.epochs * steps_per_epoch)
        if accelerator.is_main_process:
            log.info(
                f"[steps] samples_per_epoch={args.samples_per_epoch} "
                f"global_batch={args.batch_size} -> steps_per_epoch={steps_per_epoch}; "
                f"epochs={args.epochs} -> effective_max_steps={effective_max_steps} "
                f"(capped by --max-train-steps={args.max_train_steps})"
            )
    elif accelerator.is_main_process:
        log.info(f"[steps] --samples-per-epoch not set; stopping at "
                 f"--max-train-steps={effective_max_steps} (epochs ignored).")

    # ---- LR schedule --------------------------------------------------------
    # Closed-form lr(step) so it is correct regardless of --resume-step (no
    # scheduler state to checkpoint). For 'constant' this is a no-op (always
    # --ar-lr). For 'cosine': linear warmup 0 -> ar_lr over --lr-warmup-steps,
    # then cosine decay ar_lr -> --lr-min across [warmup, effective_max_steps],
    # clamped to --lr-min beyond the horizon.
    _cos_decay_steps = max(1, effective_max_steps - args.lr_warmup_steps)

    def lr_at(step: int) -> float:
        if args.lr_scheduler == "constant":
            return args.ar_lr
        if step < args.lr_warmup_steps:
            return args.ar_lr * float(step) / float(max(1, args.lr_warmup_steps))
        progress = float(step - args.lr_warmup_steps) / float(_cos_decay_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.lr_min + (args.ar_lr - args.lr_min) * cosine

    if accelerator.is_main_process and args.lr_scheduler != "constant":
        log.info(
            f"[lr] scheduler={args.lr_scheduler} peak={args.ar_lr} min={args.lr_min} "
            f"warmup={args.lr_warmup_steps} decay_steps={_cos_decay_steps} "
            f"(horizon=effective_max_steps={effective_max_steps})"
        )

    # ----- torch.compile -----------------------------------------------------
    # Compile only the AR model (the only trained module here). VQ.encode is
    # called as a method (not __call__), so OptimizedModule proxying gives
    # no speedup; leave VQ untouched.
    if args.compile:
        ar = torch.compile(ar)
        vq = torch.compile(vq)

    # NOTE: train_loader is intentionally NOT prepared (streaming, pre-sharded).
    if projectors is not None:
        ar, projectors, opt = accelerator.prepare(ar, projectors, opt)
    else:
        ar, opt = accelerator.prepare(ar, opt)

    def _orig(m):
        """Strip torch.compile's OptimizedModule wrapper if present."""
        return m._orig_mod if args.compile else m

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(copy.deepcopy(args)),
            init_kwargs={"wandb": {"name": args.exp_name}},
        )

    progress = tqdm(
        range(effective_max_steps), initial=global_step, desc="Step",
        disable=not accelerator.is_local_main_process,
    )

    # Lazily-constructed COCO FID evaluator (built on first eval, then cached
    # with its GT reference features for the rest of the run).
    coco_fid_evaluator = None

    def save_checkpoint(step):
        """Save {ar, ema, opt, (projectors)} -> {ckpt_dir}/{step:07d}.pt (rank 0)."""
        if not accelerator.is_main_process:
            return
        ck = {
            "ar": _orig(accelerator.unwrap_model(ar)).state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
            "steps": step,
        }
        if projectors is not None:
            ck["projectors"] = accelerator.unwrap_model(projectors).state_dict()
        path = f"{ckpt_dir}/{step:07d}.pt"
        torch.save(ck, path)
        log.info(f"Saved checkpoint to {path}")

    def run_coco_fid(step):
        """COCO caption->image FID + CLIPScore on the EMA AR (cfg 1.0 & cfg<S>).

        ALL ranks must call this (collective all_gather inside)."""
        nonlocal coco_fid_evaluator
        if args.coco_fid_steps <= 0:
            return
        if coco_fid_evaluator is None:
            coco_fid_evaluator = CocoFIDEvaluator(
                parquet_dir=args.coco_fid_dataset_path,
                num_samples=args.coco_fid_num_samples,
                image_size=args.coco_fid_image_size,
                seed=args.coco_fid_seed,
                inception_batch_size=args.coco_fid_inception_batch,
                clip_model_path=(args.clip_model_path or None),
                clip_batch_size=args.clip_batch_size,
            )
            coco_fid_evaluator.setup(device, torch.float32)

        eval_ar = _orig(accelerator.unwrap_model(ar))
        prev_ar_training = eval_ar.training
        # Swap in EMA params (CPU-stashed live params restored after).
        stashed = [p.detach().cpu().clone() for p in eval_ar.parameters()]
        for ema_p, mp in zip(ema.parameters(), eval_ar.parameters()):
            mp.data.copy_(ema_p.to(mp.device).data)
        eval_ar.eval()

        fid_logs = {}
        for cfg_s in [1.0, float(args.coco_fid_cfg_scale)]:
            def _gen(caption, _cfg=cfg_s):
                return ar_generate_pil(
                    eval_ar, vq, tokenizer, text_model,
                    uncond_emb_1, uncond_mask_1, caption,
                    latent_size=latent_size,
                    codebook_embed_dim=args.codebook_embed_dim,
                    text_max_len=args.text_max_len, device=device,
                    cfg_scale=_cfg,
                    temperature=args.sampling_temperature,
                    top_k=args.sampling_top_k,
                    top_p=args.sampling_top_p,
                )
            fid_val, _info = coco_fid_evaluator.evaluate(
                generate_pil_fn=_gen,
                rank=accelerator.process_index,
                world_size=accelerator.num_processes,
                verbose=True,
            )
            if accelerator.is_main_process:
                tag = "wocfg" if abs(cfg_s - 1.0) < 1e-6 else f"cfg{cfg_s:g}"
                fid_logs[f"coco_fid/{tag}"] = fid_val
                if _info.get("clip_score") is not None:
                    fid_logs[f"coco_clip/{tag}"] = _info["clip_score"]

        # Restore live AR params + training mode (also clears KV cache via train()).
        for sp, mp in zip(stashed, eval_ar.parameters()):
            mp.data.copy_(sp.to(mp.device).data)
        if prev_ar_training:
            eval_ar.train()
        else:
            eval_ar.eval()
        if accelerator.is_main_process and fid_logs:
            accelerator.log(fid_logs, step=step)

    while True:
        for batch in train_loader:
            raw_image = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            text_attn_mask = batch["attn_mask"].to(device, non_blocking=True)
            processed = preprocess_imgs_for_codec(raw_image)
            B = processed.shape[0]

            # ---- Frozen Qwen text encoding + caption dropout --------------
            # The unconditional (CFG) branch is the empty-string embedding;
            # caption dropout replaces a whole sample's text emb with it.
            with torch.no_grad(), accelerator.autocast():
                text_emb = encode_text(text_model, input_ids, text_attn_mask)
            text_mask = text_attn_mask
            if args.caption_dropout > 0.0:
                drop = torch.rand(B, device=device) < args.caption_dropout
                if drop.any():
                    text_emb = torch.where(
                        drop[:, None, None],
                        uncond_emb_1.to(text_emb.dtype).expand(B, -1, -1),
                        text_emb,
                    )
                    text_mask = torch.where(
                        drop[:, None],
                        uncond_mask_1.expand(B, -1),
                        text_mask,
                    )

            # ---- DINOv2 / target features ---------------------------------
            # Skipped entirely when REPA is disabled -- saves the DINOv2
            # forward (which would otherwise run every step). See
            # ``--repa-encoder-layer`` / ``--repa-target-spnorm`` for the
            # iREPA-style pre-processing knobs.
            if args.use_repa:
                zs = []
                with torch.no_grad(), accelerator.autocast():
                    for enc, etype in zip(encoders, encoder_types):
                        raw_in = preprocess_raw_image(raw_image, etype)
                        z_target = extract_at_layer(
                            enc, etype, raw_in, layer=args.repa_encoder_layer,
                        )
                        z_target = spatial_norm(
                            z_target,
                            mode=args.repa_target_spnorm,
                            alpha=args.repa_spnorm_alpha,
                        )
                        zs.append(z_target)

            # ---- Frozen VQ encode -> hard token indices --------------------
            with torch.no_grad():
                _, _, [_, _, indices] = accelerator.unwrap_model(vq).encode(processed)
            z_indices = indices.reshape(B, -1)  # (B, L)

            ar.train()
            accum_modules = [ar] if projectors is None else [ar, projectors]
            with accelerator.accumulate(accum_modules), accelerator.autocast():
                if args.use_repa:
                    logits, hidden_at_tap = ar(
                        idx=z_indices[:, :-1], cond_emb=text_emb, text_mask=text_mask,
                        return_hidden_at=args.encoder_depth,
                    )
                else:
                    logits = ar(idx=z_indices[:, :-1], cond_emb=text_emb, text_mask=text_mask)
                ce_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    z_indices.reshape(-1),
                )
                if args.use_repa:
                    zs_tilde = project_repa(hidden_at_tap, args.cls_token_num, projectors)
                    proj_loss_ar = repa_align_loss(zs_tilde, zs)
                    ar_total = ce_loss + args.proj_coeff * proj_loss_ar
                else:
                    proj_loss_ar = None
                    ar_total = ce_loss

                accelerator.backward(ar_total)
                grad_norm_ar = None
                if accelerator.sync_gradients:
                    clip_params = list(ar.parameters())
                    if projectors is not None:
                        clip_params = clip_params + list(projectors.parameters())
                    grad_norm_ar = accelerator.clip_grad_norm_(
                        clip_params, args.max_grad_norm,
                    )
                    # Set this optimizer step's LR from the closed-form schedule
                    # (constant => unchanged --ar-lr; cosine => warmup/decay).
                    cur_lr = lr_at(global_step)
                    for pg in opt.param_groups:
                        pg["lr"] = cur_lr
                opt.step(); opt.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, _orig(accelerator.unwrap_model(ar)))

            if accelerator.sync_gradients:
                progress.update(1); global_step += 1

                # Log keys deliberately match Stage 1's keys (`ce_loss`,
                # `proj_loss_ar`, `ar_total`, `grad_norm_ar`) so the same
                # wandb panel can overlay both stages for comparison.
                logs = {
                    "ce_loss": accelerator.gather(ce_loss).mean().item(),
                    "ar_total": accelerator.gather(ar_total).mean().item(),
                }
                if proj_loss_ar is not None:
                    logs["proj_loss_ar"] = accelerator.gather(proj_loss_ar).mean().item()
                if grad_norm_ar is not None:
                    logs["grad_norm_ar"] = accelerator.gather(grad_norm_ar).mean().item()
                progress.set_postfix(**{k: f"{v:.3f}" for k, v in logs.items()})
                accelerator.log(logs, step=global_step)

            if accelerator.sync_gradients and (global_step % args.checkpointing_steps == 0 and global_step > 0):
                save_checkpoint(global_step)

            # ---- recon + AR sample logging ---------------------------------
            # Visualisation only -- a handful of images for sanity checking.
            # Done ENTIRELY on the main rank (no gather, other ranks idle).
            if accelerator.sync_gradients and (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                if accelerator.is_main_process and args.report_to == "wandb":
                    unwrapped_vq = vq  # not wrapped (frozen, skipped prepare())
                    # Use the original (uncompiled) AR for ``generate()`` --
                    # repeated KV-cache mutations would otherwise force the
                    # OptimizedModule wrapper to recompile every sampling step.
                    unwrapped_ar = _orig(accelerator.unwrap_model(ar))
                    unwrapped_ar.eval()

                    # (1) Reconstruction pairs (input | recon side-by-side) using
                    # the *frozen* Stage-1 VQ -- sanity-checks the H5 dataset /
                    # preprocessing path matches what the AR was trained against.
                    n_recon = max(1, min(args.sampling_num, processed.shape[0]))
                    with torch.no_grad():
                        inp_log = processed[:n_recon]
                        recon_log = unwrapped_vq(inp_log)[0]
                        recon_log = (recon_log.clamp(-1, 1) + 1) / 2.0
                        inp_log = (inp_log.clamp(-1, 1) + 1) / 2.0

                    # (2) AR-generated samples from fixed prompts. NOTE: do NOT
                    # wrap in ``accelerator.autocast()`` -- ``setup_caches``
                    # allocates KVCache buffers in ``tok_embeddings.weight.dtype``
                    # (fp32 under accelerate mixed-precision since params stay
                    # fp32), and bf16 activations would clash with the fp32
                    # in-place ``k_cache[..., input_pos] = k_val`` write.
                    prompts = SAMPLE_PROMPTS[:max(1, args.sampling_num)]
                    cond_emb_s, cond_mask_s = encode_prompts(
                        tokenizer, text_model, prompts, args.text_max_len, device,
                    )
                    n_sample = cond_emb_s.shape[0]
                    # Cast text embeddings to the AR param dtype (fp32 under
                    # accelerate mixed precision) since generate runs without
                    # autocast (KV cache inherits tok_embeddings dtype).
                    ar_dtype = next(unwrapped_ar.parameters()).dtype
                    cond_emb_s = cond_emb_s.to(ar_dtype)
                    uncond_emb_s = uncond_emb_1.to(ar_dtype).expand(n_sample, -1, -1)
                    uncond_mask_s = uncond_mask_1.expand(n_sample, -1)
                    index_sample = generate_t2i(
                        unwrapped_ar, cond_emb_s, uncond_emb_s, latent_size ** 2,
                        cond_mask=cond_mask_s, uncond_mask=uncond_mask_s,
                        cfg_scale=args.sampling_cfg_scale,
                        cfg_interval=args.sampling_cfg_interval,
                        temperature=args.sampling_temperature,
                        top_k=args.sampling_top_k,
                        top_p=args.sampling_top_p,
                        sample_logits=True,
                    )
                    qzshape = [
                        n_sample,
                        args.codebook_embed_dim,
                        latent_size, latent_size,
                    ]
                    with torch.no_grad():
                        sample_imgs = unwrapped_vq.decode_code(index_sample, qzshape)
                        sample_imgs = (sample_imgs.clamp(-1, 1) + 1) / 2.0

                    # Drop kv-cache state (next iter's training-mode forward
                    # would otherwise hit the cached K/V and crash).
                    unwrapped_ar.train()

                    # Log each AR sample as its own wandb.Image so the prompt
                    # shows up as the image caption (a single grid would lose
                    # the per-image text association).
                    ar_sample_imgs = []
                    sample_imgs_cpu = sample_imgs.float().clamp(0, 1).cpu()
                    for i, prompt in enumerate(prompts):
                        arr = (sample_imgs_cpu[i] * 255).round().to(torch.uint8)
                        arr = arr.permute(1, 2, 0).numpy()  # (H, W, 3)
                        ar_sample_imgs.append(wandb.Image(arr, caption=prompt))

                    accelerator.log(
                        {
                            "recon_grid": wandb.Image(
                                array2grid_pairs(
                                    inp_log.float(), recon_log.float(),
                                    pairs_per_row=args.sampling_pairs_per_row,
                                )
                            ),
                            "ar_samples": ar_sample_imgs,
                        },
                        step=global_step,
                    )

            # ---- COCO FID eval (t2i) ---------------------------------------
            # ImageNet class-conditional FID is meaningless for t2i, so we
            # instead compute caption->image FID against a COCO subset. We
            # evaluate the EMA AR (smoother inference proxy) at two settings:
            #   * coco_fid/wocfg   -- cfg_scale = 1.0 (no CFG)
            #   * coco_fid/cfg<S>  -- cfg_scale = args.coco_fid_cfg_scale (1.5)
            # ALL ranks must enter this block (collective all_gather inside),
            # so it is gated only by the step schedule, not by rank.
            # ---- COCO FID eval (t2i) ---------------------------------------
            # caption->image FID + CLIPScore on the EMA AR (cfg 1.0 & cfg<S>).
            # ALL ranks must enter run_coco_fid (collective all_gather inside).
            if (
                args.coco_fid_steps > 0
                and accelerator.sync_gradients
                and (global_step % args.coco_fid_steps == 0 and global_step > 0)
            ):
                run_coco_fid(global_step)

            if global_step >= effective_max_steps:
                break
        # The streaming dataset cycles forever, so the for-loop only ends via
        # the step-budget break above (or an empty-shard misconfig); stop.
        break

    # ---- final save + eval -------------------------------------------------
    # Training usually ends at a step that is NOT a multiple of
    # checkpointing/eval intervals (e.g. ~390k for GPIC 1 epoch), which would
    # otherwise discard the last partial interval. Do one final save + COCO FID
    # here, skipping it only if this exact step was already saved/evaluated.
    if global_step > 0 and (global_step % args.checkpointing_steps != 0):
        save_checkpoint(global_step)
    if (
        args.coco_fid_steps > 0
        and global_step > 0
        and (global_step % args.coco_fid_steps != 0)
    ):
        run_coco_fid(global_step)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        log.info("Stage 2 done.")
    accelerator.end_training()


def parse_args():
    p = argparse.ArgumentParser()

    # logging
    p.add_argument("--output-dir", type=str, default="exps")
    p.add_argument("--exp-name", type=str, required=True)
    p.add_argument("--logging-dir", type=str, default="logs")
    p.add_argument("--report-to", type=str, default="wandb")
    p.add_argument("--wandb-project", type=str, default="src_stage2")
    p.add_argument("--sampling-steps", type=int, default=20000)
    # AR sampling controls (mirror Stage 1 / inference.py demo defaults)
    p.add_argument("--sampling-num", type=int, default=4,
                   help="Number of recon pairs / AR samples to log to wandb "
                        "at each sampling step. Computed on the main rank "
                        "only -- no gather across ranks.")
    p.add_argument("--sampling-pairs-per-row", type=int, default=4,
                   help="Pairs (input|recon) per row in recon_grid.")
    p.add_argument("--sampling-cfg-scale", type=float, default=4.0)
    p.add_argument("--sampling-cfg-interval", type=int, default=-1)
    p.add_argument("--sampling-temperature", type=float, default=1.0)
    p.add_argument("--sampling-top-k", type=int, default=2000)
    p.add_argument("--sampling-top-p", type=float, default=1.0)
    p.add_argument("--resume-step", type=int, default=0)

    # Online COCO caption->image FID eval (t2i). ImageNet class-conditional
    # FID is meaningless here, so we compute FID against a COCO subset.
    p.add_argument("--coco-fid-steps", type=int, default=0,
                   help="Run a COCO-caption FID eval every N global steps. "
                        "Set <= 0 (default) to disable. Evaluates the EMA AR at "
                        "cfg_scale=1.0 (wo cfg) and --coco-fid-cfg-scale.")
    p.add_argument("--coco-fid-dataset-path", type=str, default=None,
                   help="Directory of COCO parquet files with columns "
                        "`caption` and `image` (HF Image struct). Required when "
                        "--coco-fid-steps > 0.")
    p.add_argument("--coco-fid-num-samples", type=int, default=1000,
                   help="Number of COCO captions sampled (deterministic by seed). "
                        "Higher = more stable FID but linearly more generation.")
    p.add_argument("--coco-fid-image-size", type=int, default=256,
                   help="Square size for GT + generated PILs before InceptionV3.")
    p.add_argument("--coco-fid-seed", type=int, default=42,
                   help="Seed for the deterministic COCO caption subset selection.")
    p.add_argument("--coco-fid-inception-batch", type=int, default=32,
                   help="InceptionV3 feature-extraction batch size.")
    p.add_argument("--coco-fid-cfg-scale", type=float, default=1.5,
                   help="The 'with CFG' scale for the second COCO FID pass "
                        "(the first pass is always cfg_scale=1.0).")
    p.add_argument("--clip-model-path", type=str,
                   default="openai/clip-vit-base-patch32",
                   help="CLIP model for CLIPScore (image-text alignment of the "
                        "generated samples), computed in the same COCO FID pass. "
                        "Empty disables CLIPScore. Reported as 100*mean cosine "
                        "(typically ~20-30+). Logged under coco_clip/{wocfg,cfg<S>}.")
    p.add_argument("--clip-batch-size", type=int, default=64,
                   help="Batch size for CLIP image/text feature extraction.")

    # Legacy class-conditional FID args (kept for CLI compat; unused for t2i).
    p.add_argument("--eval-steps", type=int, default=0,
                   help="[unused for t2i] legacy class-conditional FID cadence.")
    p.add_argument("--eval-fid-num", type=int, default=50000,
                   help="Number of samples used for FID. Canonical 50_000; "
                        "10_000 is a faster online proxy.")
    p.add_argument("--eval-per-proc-batch-size", type=int, default=32,
                   help="Per-rank batch size during FID sampling.")
    p.add_argument("--fid-reference-file", type=str, default="",
                   help="Path to precomputed reference statistics (.npz "
                        "containing `mu`/`sigma`). Empty disables FID.")
    p.add_argument("--eval-temperature", type=float, default=1.0,
                   help="Temperature used by `generate()` during FID eval.")
    p.add_argument("--eval-top-k", type=int, default=0,
                   help="top-k for `generate()` during FID eval (0 = no top-k).")
    p.add_argument("--eval-top-p", type=float, default=1.0,
                   help="top-p for `generate()` during FID eval (1.0 = no top-p).")
    p.add_argument("--eval-fid-batch-size", type=int, default=200,
                   help="InceptionV3 batch size during FID statistics computation.")
    p.add_argument("--eval-fid-num-workers", type=int, default=8,
                   help="DataLoader workers for InceptionV3 features.")
    p.add_argument("--eval-save-workers", type=int, default=8,
                   help="Per-rank thread-pool size for async PNG encode+write "
                        "during FID sampling. See train_gear.py for details.")
    p.add_argument("--eval-keep-samples", action="store_true", default=False,
                   help="Keep generated PNGs after FID is computed (default: delete).")

    # data (WebDataset, text-to-image)
    p.add_argument("--data-type", type=str, default="blip3o",
                   choices=["blip3o", "gpic"],
                   help="WebDataset layout of --data-dir: 'blip3o' = {key}.jpg + "
                        "{key}.txt; 'gpic' = {key}.json (caption inside) + "
                        "{key}.jpg/.png. Selects the row extractor.")
    p.add_argument("--data-dir", type=str, required=True,
                   help="WebDataset source: a directory of .tar shards, a glob "
                        "(e.g. /path/*.tar), or a single .tar. Comma-separate "
                        "multiple sources for interleaved streaming.")
    p.add_argument("--image-size", type=int, default=256,
                   help="Short-side resize target then center-crop (256 or 512).")
    p.add_argument("--num-classes", type=int, default=1000,
                   help="Unused for t2i conditioning; kept for ModelArgs compat.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--random-hflip", action=argparse.BooleanOptionalAction, default=True,
                   help="Apply 50%% random horizontal flip in the data pipeline. "
                        "Standard for image generation training -- disable with "
                        "--no-random-hflip for ablation runs.")

    # text encoder (Qwen3) for t2i conditioning
    p.add_argument("--text-encoder", type=str,
                   default="Qwen/Qwen3-1.7B",
                   help="Path to the frozen Qwen text encoder. Works with a plain "
                        "text LLM (Qwen3-1.7B, hidden=2048) or a VLM whose text "
                        "tower is under .language_model (Qwen3.5). The AR "
                        "caption_dim is set to the text encoder hidden size.")
    p.add_argument("--text-max-len", type=int, default=300,
                   help="Fixed caption token length (right-padded). Becomes the AR "
                        "text-prefix length; must equal --cls-token-num.")
    p.add_argument("--caption-dropout", type=float, default=0.1,
                   help="Probability of replacing a sample's caption embedding with "
                        "the empty-string (uncond) embedding, enabling CFG.")
    p.add_argument("--data-probs", type=str, default="",
                   help="Comma-separated per-source interleave weights, in the SAME "
                        "order as --data-dir sources (normalised internally). Pass "
                        "the dataset sizes (e.g. '27157092,4773675') so the sources "
                        "deplete proportionally and ~1 epoch == each sample seen "
                        "~once. Empty => equal probability (round-robin).")

    # frozen VQ
    p.add_argument("--vq-model", type=str, default="VQ-16",
                   help="Tokenizer family / size. Looked up in "
                        "models.Tokenizers (currently: VQ-8 / VQ-16 / LFQ-16 / "
                        "IBQ-16). For LFQ-16 the codebook_embed_dim is forced "
                        "to log2(codebook_size) regardless of the CLI value; "
                        "for IBQ-16 use --codebook-embed-dim=256.")
    p.add_argument("--codebook-size", type=int, default=16384)
    p.add_argument("--codebook-embed-dim", type=int, default=8)
    p.add_argument("--downsample-ratio", type=int, default=16)
    p.add_argument("--vq-ckpt", type=str, required=True)
    p.add_argument(
        "--vq-use-ema", action=argparse.BooleanOptionalAction, default=True,
        help="When --vq-ckpt is one of our Stage-0/1 ckpts (which carry "
             "both `vq` and `vq_ema`), load the EMA shadow instead of the "
             "live params. EMA is what Stage 1 reports under `vq_val_ema/*` "
             "and is the recommended frozen tokenizer for Stage 2. Falls "
             "back silently to the live `vq` if `vq_ema` is absent. "
             "Has no effect on legacy `{model: ...}` ckpts (e.g. "
             "`vq_ds16_c2i.pt`) or MAGVIT2 lightning ckpts (those expose "
             "only one tensor source).",
    )

    # AR
    p.add_argument("--ar-model", type=str, default="LlamaGen-B",
                   choices=list(LlamaGen_models.keys()))
    p.add_argument("--ar-init-ckpt", type=str, default=None,
                   help="Optional c2i (or t2i) checkpoint to warm-start the AR. "
                        "A c2i ckpt loads into the image stream (strict=False).")
    p.add_argument("--cls-token-num", type=int, default=300,
                   help="Text-prefix length; must equal --text-max-len for t2i.")
    p.add_argument("--text-attn", type=str, default="causal",
                   choices=["causal", "prefix"],
                   help="Attention topology over the text prefix. 'causal' "
                        "(default): the whole [text; image] sequence is a plain "
                        "causal LM (text token i sees text<=i). 'prefix': text "
                        "attends bidirectionally among valid text tokens "
                        "(joint self-attention), image stays causal. Image is "
                        "always causal and the same setting MUST be used at "
                        "inference (it is saved into the ckpt args and read by "
                        "inference_t2i.py / the in-trainer sampler).")
    p.add_argument("--text-stream-copy-init", action=argparse.BooleanOptionalAction, default=True,
                   help="After loading a c2i init ckpt, warm-copy the image-stream "
                        "attention params (wqkv/wo/attention_norm) into the new "
                        "text stream. Disable with --no-text-stream-copy-init.")
    p.add_argument("--dropout-p", type=float, default=0.1)
    p.add_argument("--token-dropout-p", type=float, default=0.1)
    p.add_argument("--drop-path-rate", type=float, default=0.0)
    p.add_argument("--use-checkpoint", action="store_true", default=False)

    # REPA
    p.add_argument("--use-repa", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable the REPA alignment auxiliary loss. With "
                        "--no-use-repa the AR is trained with CE only -- no "
                        "DINOv2 forward, no projector, no extra loss. The "
                        "--enc-type / --encoder-depth / --projector-dim / "
                        "--proj-coeff flags are ignored in that case.")
    p.add_argument("--enc-type", type=str, default="dinov2-vit-b")
    p.add_argument("--encoder-depth", type=int, default=8)
    p.add_argument(
        "--repa-encoder-layer", type=int, default=-1,
        help=(
            "Which transformer block of the REPA encoder to align to. "
            "-1 (default) = the standard ``forward_features`` output, i.e. last "
            "block + final LayerNorm. 1..N = the n-th block's PRE-norm patch "
            "tokens (DINOv2/v3/SigLIP2/V-JEPA 2.1 ViT-B all have N=12; ViT-L has N=24)."
        ),
    )
    p.add_argument(
        "--repa-target-spnorm", type=str, default="none",
        choices=["none", "demean", "zscore"],
        help=(
            "iREPA-style per-channel spatial normalisation on the REPA target "
            "before the cosine align loss. 'demean' = z - alpha * mean_l z; "
            "'zscore' = also divide by per-channel spatial std. Default: none."
        ),
    )
    p.add_argument(
        "--repa-spnorm-alpha", type=float, default=0.6,
        help=(
            "Mean-subtraction strength for --repa-target-spnorm. iREPA's LDM "
            "uses 0.6 (default); JiT uses 0.8; alpha=1.0 makes per-channel "
            "spatial mean exactly zero."
        ),
    )
    p.add_argument("--projector-dim", type=int, default=2048)
    p.add_argument("--proj-coeff", type=float, default=0.5)

    # optim
    p.add_argument("--ar-lr", type=float, default=1e-4)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--adam-weight-decay", type=float, default=0.0)
    p.add_argument("--adam-epsilon", type=float, default=1e-8)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lr-scheduler", type=str, default="constant",
                   choices=["constant", "cosine"],
                   help="LR schedule for the AR optimizer. 'constant' (default): "
                        "fixed --ar-lr the whole run (original behaviour). "
                        "'cosine': linear warmup over --lr-warmup-steps from 0 to "
                        "--ar-lr, then cosine decay from --ar-lr down to --lr-min "
                        "over the remaining steps (horizon = effective_max_steps, "
                        "i.e. min(epochs*steps_per_epoch, --max-train-steps)). "
                        "Computed in closed form from global_step, so it resumes "
                        "correctly from --resume-step without extra state.")
    p.add_argument("--lr-warmup-steps", type=int, default=0,
                   help="Linear warmup steps for --lr-scheduler=cosine (0 = no "
                        "warmup). Ignored for the constant schedule.")
    p.add_argument("--lr-min", type=float, default=0.0,
                   help="Floor learning rate the cosine schedule decays to at "
                        "effective_max_steps (and stays at afterwards). e.g. set "
                        "to 0.1*--ar-lr for a 10x decay, or 0.0 to fully anneal. "
                        "Ignored for the constant schedule.")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--max-train-steps", type=int, default=1500000)
    p.add_argument("--epochs", type=int, default=1,
                   help="Number of dataset passes. Only takes effect together with "
                        "--samples-per-epoch (streaming length is otherwise "
                        "unknown). Effective stop = min(epochs*steps_per_epoch, "
                        "--max-train-steps).")
    p.add_argument("--samples-per-epoch", type=int, default=0,
                   help="Total images in one epoch (e.g. dataset size). Used with "
                        "--epochs to derive the step budget: "
                        "steps_per_epoch = ceil(samples_per_epoch / global_batch). "
                        "0 (default) => ignore epochs, stop at --max-train-steps.")
    p.add_argument("--checkpointing-steps", type=int, default=50000)

    # misc
    p.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="Apply torch.compile (inductor) to the AR model. EMA, "
                        "ckpt save and AR sampling reach through _orig_mod, "
                        "so behaviour is identical with --no-compile.")
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
