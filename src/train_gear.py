"""Stage 1 of the src pipeline: jointly train VQ + AR with REPA alignment.

Inspired by `REPA-E/train_repae.py`. Per training step we run **three**
optimizer updates:

1. **VQ generator step** (`opt_vq`):
   reconstruction loss + perceptual loss + GAN gen loss + VQ losses
   (vq + commit + entropy) + ``vae_align_proj_coeff * REPA_align_loss``.
   The REPA loss flows through the *non-detached* soft label
   ``softmax(-d / T)`` back into the VQ encoder and codebook (REPA-E
   spirit). Optionally, ``--vq-align-straight-through`` uses a hard
   one-hot value in the AR forward while keeping the soft-label
   gradient. AR + projector params have ``requires_grad=False`` here so
   they pass gradients through but are not updated.

2. **Discriminator step** (`opt_disc`):
   hinge loss on the cached ``recon`` (gen frozen).

3. **AR step** (`opt_ar`, AR + projectors together):
   we recompute the VQ outputs **inside `torch.no_grad()`**, then build
   the AR's input as ``softmax(-d / T)`` (which carries no gradient back
   to VQ) and run

       ar_input_soft  : (B, L, K) soft posterior            [no grad to VQ]
       logits, hidden = ar(ar_input_soft[:, :-1, :], cond_idx=labels,
                           return_hidden_at=encoder_depth)
       ce_loss   = CE(logits, hard_indices)
       repa_loss = align(MLP(hidden[:, cls_token_num-1:]), DINOv2 patches)

   Result: AR + projectors are updated by CE + REPA, while the VQ branch
   receives no gradient at all from this step.

The REPA target is a single DINOv2-ViT-B encoder by default (overridable
via ``--enc-type``).
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
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid
from tqdm.auto import tqdm

# Make the parent package importable when launching from inside src/
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from models import Tokenizers as VQ_models  # noqa: E402
# Unified registry that holds the VQ (models/vq_model.py), LFQ
# (models/lfq_model.py) and IBQ (models/ibq_model.py) families. Aliased
# back to ``VQ_models`` so the existing ``VQ_models[args.vq_model]`` lookup
# is unchanged -- pass ``--vq-model=LFQ-16`` / ``IBQ-16`` (or ``VQ-16`` /
# ``VQ-8``) at the CLI to switch.
from models.llamagen import LlamaGen_models  # noqa: E402  - native UniWorld AR
from models.generate import generate  # noqa: E402  - AR sampling (CFG + top-k/top-p)

from src.dataset import (  # noqa: E402
    build_imagenet_dataset,
    build_imagenet_val_dataset,
)
from src.losses import ReconstructionLossVQ  # noqa: E402
from src.utils import (  # noqa: E402
    count_trainable_params,
    extract_at_layer,
    extract_repa_target,
    get_constant_schedule_with_warmup,
    load_encoders,
    load_pretrained_tokenizer_state_dict,
    num_encoder_layers,
    preprocess_imgs_for_codec,
    preprocess_raw_image,
    run_distributed_fid_eval,
    run_vq_reconstruction_eval,
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
    """Lay out (input, recon) pairs side-by-side in a grid.

    Each pair occupies two adjacent cells: input on the left, recon on its
    right. ``pairs_per_row`` pairs are placed per row, giving rows of width
    ``2 * pairs_per_row`` cells. Both tensors must already be in [0, 1].
    """
    assert inputs.shape == recons.shape, (
        f"inputs.shape {tuple(inputs.shape)} != recons.shape {tuple(recons.shape)}"
    )
    pairs = torch.stack([inputs, recons], dim=1).reshape(-1, *inputs.shape[1:])
    return array2grid(pairs, nrow=2 * pairs_per_row)


def build_repa_projector(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    """3-layer SiLU MLP, same shape as REPA-E projector."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class ProjectorBank(nn.Module):
    """A fixed set of REPA projectors that DDP can wrap as a single module.

    ``forward(patch_hidden)`` returns a ``list[Tensor]`` -- one projection
    per encoder. We need this wrapper because ``accelerator.prepare()`` wraps
    its arguments in ``DistributedDataParallel`` and a DDP-wrapped
    ``nn.ModuleList`` is not iterable. With this bank, the call site does
    ``projectors(patch_hidden)`` (DDP forward path) instead of iterating.
    """

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
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())
    for name, buffer in model_buffers.items():
        name = name.replace("module.", "")
        if buffer.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            ema_buffers[name].mul_(decay).add_(buffer.data, alpha=1 - decay)
        else:
            ema_buffers[name].copy_(buffer)


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
    )
    return logging.getLogger(__name__)


def requires_grad(model_or_iter, flag=True):
    if hasattr(model_or_iter, "parameters"):
        params = model_or_iter.parameters()
    else:
        params = model_or_iter
    for p in params:
        p.requires_grad = flag


def repa_align_loss(zs_tilde_list, zs_list):
    """Mean (over batch and tokens) of ``-cos(zs_tilde, zs)``."""
    proj = torch.zeros((), device=zs_tilde_list[0].device)
    bsz = zs_list[0].shape[0]
    for z, z_tilde in zip(zs_list, zs_tilde_list):
        for z_j, z_tilde_j in zip(z, z_tilde):
            z_tilde_j = F.normalize(z_tilde_j, dim=-1)
            z_j = F.normalize(z_j, dim=-1)
            proj = proj + (-(z_j * z_tilde_j).sum(dim=-1)).mean()
    return proj / (len(zs_list) * bsz)


# (encode helper not needed: ``vq(input, return_distance=True)`` returns
# ``(recon, z_q, vq_losses_tuple, info_tuple, d)`` in a single DDP forward.)


def project_repa(hidden_at_tap, cls_token_num, projectors):
    """Slice cls_token tail from hidden tap and run the projector bank.

    ``hidden_at_tap`` shape : (B, cls_token_num + L - 1, dim_ar).
    We take the last L positions (i.e. ``hidden_at_tap[:, cls_token_num - 1:]``
    so that index k of the result corresponds to image patch k -- mirrors
    how the original Transformer slices logits at training time).

    ``projectors`` is a :class:`ProjectorBank` (possibly DDP-wrapped); calling
    it directly goes through DDP's forward so gradient sync is preserved.
    """
    patch_hidden = hidden_at_tap[:, cls_token_num - 1:]  # (B, L, D)
    return projectors(patch_hidden)


# =============================================================================
# Main
# =============================================================================
def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )

    save_dir = os.path.join(args.output_dir, args.exp_name)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        log = create_logger(save_dir)
        log.info(f"Experiment directory created at {save_dir}")

    device = accelerator.device
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # =========================================================================
    # 1. Build native VQ
    # =========================================================================
    vq = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    )
    if args.vq_pretrained_ckpt:
        # Same helper handles native ``{"vq": ...}`` / legacy ``{"model": ...}``
        # / TencentARC MAGVIT2 lightning ckpts (LFQ) -- so a single
        # --vq-pretrained-ckpt flag works for both VQ-16 and LFQ-16 runs.
        sd = load_pretrained_tokenizer_state_dict(args.vq_pretrained_ckpt)
        missing, unexpected = vq.load_state_dict(sd, strict=False)
        if accelerator.is_main_process:
            log.info(f"Loaded VQ from {args.vq_pretrained_ckpt} | missing={len(missing)}, unexpected={len(unexpected)}")
    if args.entropy_loss_ratio is not None:
        vq.quantize.entropy_loss_ratio = float(args.entropy_loss_ratio)
    if args.commit_loss_beta is not None:
        # See train_tokenizer.py for the rationale (VQ uses ``beta``, LFQ
        # uses ``commit_loss_beta``; one flag overrides whichever exists).
        if hasattr(vq.quantize, "commit_loss_beta"):
            vq.quantize.commit_loss_beta = float(args.commit_loss_beta)
        elif hasattr(vq.quantize, "beta"):
            vq.quantize.beta = float(args.commit_loss_beta)
    # ``VectorQuantizer.codebook_used`` (a per-rank sliding-window buffer)
    # has been removed: it caused dynamo graph breaks (data-dependent
    # ``len(torch.unique(...))``) and AOTAutograd saved-tensor lifetime
    # issues under our multi-backward-per-iter pattern. We compute global
    # ``codebook_usage`` ourselves in the training loop via
    # ``accelerator.gather(indices)``.
    vq = vq.to(device)

    # =========================================================================
    # 2. Build native AR (LlamaGen) + REPA projectors
    # =========================================================================
    assert args.image_size % args.downsample_ratio == 0, \
        "Image size must be divisible by VQ downsample ratio."
    latent_size = args.image_size // args.downsample_ratio
    block_size = latent_size ** 2

    encoders, encoder_types, _ = load_encoders(args.enc_type, device, args.image_size)
    z_dims = [enc.embed_dim for enc in encoders]
    if args.repa_encoder_layer != -1:
        # Validate the requested block index against every encoder up front
        # so we fail loudly at startup instead of mid-training.
        for enc, etype in zip(encoders, encoder_types):
            n_blk = num_encoder_layers(enc, etype)
            if not (1 <= args.repa_encoder_layer <= n_blk):
                raise ValueError(
                    f"--repa-encoder-layer={args.repa_encoder_layer} is out of "
                    f"range for {etype} (has {n_blk} blocks; valid: -1 or "
                    f"1..{n_blk})."
                )

    ar = LlamaGen_models[args.ar_model](
        block_size=block_size,
        vocab_size=args.codebook_size,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        resid_dropout_p=args.dropout_p,
        ffn_dropout_p=args.dropout_p,
        token_dropout_p=args.token_dropout_p,
        drop_path_rate=args.drop_path_rate,
        use_checkpoint=args.use_checkpoint,
    ).to(device)
    if not (1 <= args.encoder_depth <= len(ar.layers)):
        raise ValueError(f"--encoder-depth={args.encoder_depth} out of range [1, {len(ar.layers)}].")

    projectors = ProjectorBank(
        in_dim=ar.config.dim, hidden=args.projector_dim, out_dims=z_dims,
    ).to(device)

    ema = copy.deepcopy(ar).to(device)
    requires_grad(ema, False)

    # VQ EMA: a frozen, exponentially-averaged copy of the live VQ. Stage 1
    # trains the VQ end-to-end so the live params bounce around with each
    # step (REPA pulls the codebook, GAN gradients are noisy, etc.). The
    # EMA gives a smoother, more useful weight for downstream eval / FID
    # / sampling. Built BEFORE accelerator.prepare so it stays a plain
    # local nn.Module (not DDP-wrapped) and is updated by hand after every
    # VQ optimizer step. requires_grad=False everywhere -- pure storage.
    vq_ema = copy.deepcopy(vq).to(device)
    requires_grad(vq_ema, False)
    vq_ema.eval()

    # =========================================================================
    # 3. Loss + discriminator
    # =========================================================================
    loss_cfg = OmegaConf.load(args.loss_cfg_path)
    vq_loss_fn = ReconstructionLossVQ(loss_cfg).to(device)
    if args.disc_pretrained_ckpt:
        disc_state = torch.load(args.disc_pretrained_ckpt, map_location=device)
        vq_loss_fn.discriminator.load_state_dict(disc_state)
        if accelerator.is_main_process:
            log.info(f"Loaded discriminator init from {args.disc_pretrained_ckpt}")

    # ----- Stage-0 init -------------------------------------------------------
    # Single-flag warm start from a Stage-0 checkpoint (``train_tokenizer.py``):
    # loads ``vq`` + ``vq_ema`` + ``discriminator`` from the same file. Done
    # AFTER ``--vq-pretrained-ckpt`` / ``--disc-pretrained-ckpt`` so it
    # overrides them if both are passed (with a warning). Stage-1 builds
    # ``vq_ema`` further up as ``deepcopy(vq)``; here we overwrite that copy
    # with the EMA tensor stored in the Stage-0 ckpt (keeps the smoother
    # weight rather than restarting EMA from the live VQ).
    #
    # Optimizer / scheduler state is intentionally NOT migrated: Stage 1 has
    # a different loss recipe (REPA gradient enters the VQ branch on top of
    # the GAN/recon losses), so reusing Adam's second moment statistics
    # across stages is more likely to destabilise than to help.
    if args.stage0_init_ckpt:
        if args.vq_pretrained_ckpt or args.disc_pretrained_ckpt:
            if accelerator.is_main_process:
                log.info(
                    "Note: --stage0-init-ckpt is set; it overrides "
                    "--vq-pretrained-ckpt and --disc-pretrained-ckpt."
                )
        s0 = torch.load(args.stage0_init_ckpt, map_location="cpu")
        miss_v, unexp_v = vq.load_state_dict(s0["vq"], strict=False)
        if "vq_ema" in s0:
            vq_ema.load_state_dict(s0["vq_ema"], strict=False)
            ema_src = "ckpt[vq_ema]"
        else:
            vq_ema.load_state_dict(s0["vq"], strict=False)
            ema_src = "ckpt[vq] (no vq_ema in ckpt)"
        vq_loss_fn.discriminator.load_state_dict(s0["discriminator"])
        if accelerator.is_main_process:
            log.info(
                f"Loaded Stage-0 init from {args.stage0_init_ckpt} "
                f"(stage={s0.get('stage', '?')}, steps={s0.get('steps', '?')}) | "
                f"vq missing={len(miss_v)}, unexpected={len(unexp_v)}; "
                f"vq_ema <- {ema_src}; discriminator=loaded"
            )

    # ``NLayerDiscriminator`` uses plain ``nn.BatchNorm2d``; without sync the
    # per-rank running stats diverge across ranks (each rank sees only a
    # local mini-batch), which biases the discriminator output. Mirror
    # REPA-E: convert the discriminator's BNs to SyncBatchNorm so all ranks
    # see the same global statistics. Must happen *before* prepare so the
    # converted module gets wrapped by DDP.
    if accelerator.use_distributed:
        vq_loss_fn.discriminator = (
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(vq_loss_fn.discriminator)
        )

    if accelerator.is_main_process:
        log.info(f"AR params       : {sum(p.numel() for p in ar.parameters()):,}")
        log.info(f"Projector params: {sum(p.numel() for p in projectors.parameters()):,}")
        log.info(f"VQ params       : {sum(p.numel() for p in vq.parameters()):,}")
        log.info(f"Trainable VQ    : {count_trainable_params(vq):,}")
        log.info(f"Trainable AR    : {count_trainable_params(ar):,}")

    # =========================================================================
    # 4. Optimizers
    # =========================================================================
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    opt_ar = torch.optim.AdamW(
        list(ar.parameters()) + list(projectors.parameters()),
        lr=args.ar_lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )
    opt_vq = torch.optim.AdamW(
        vq.parameters(), lr=args.vq_lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )
    opt_disc = torch.optim.AdamW(
        vq_loss_fn.parameters(), lr=args.disc_lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )

    # Linear warmup -> constant LR for all three optimizers. Same warmup
    # length is applied to AR / VQ / disc -- they're all on the same step
    # clock and keeping a single knob avoids three independently-tuned
    # transients. With ``--warmup-steps 1``, the first optimizer.step()
    # runs at lr=0 (no parameter update), so an eval at ``global_step==1``
    # shows the *pretrained* baseline. With ``--warmup-steps 0``, lr stays
    # at base from step 0 (no warmup).
    sched_ar = get_constant_schedule_with_warmup(opt_ar, num_warmup_steps=args.warmup_steps)
    sched_vq = get_constant_schedule_with_warmup(opt_vq, num_warmup_steps=args.warmup_steps)
    sched_disc = get_constant_schedule_with_warmup(opt_disc, num_warmup_steps=args.warmup_steps)

    # =========================================================================
    # 5. Data
    # =========================================================================
    train_dataset = build_imagenet_dataset(
        args.data_dir, image_size=args.image_size, random_hflip=args.random_hflip,
    )
    local_bs = int(args.batch_size // accelerator.num_processes)
    train_loader = DataLoader(
        train_dataset, batch_size=local_bs, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    if accelerator.is_main_process:
        log.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")

    # =========================================================================
    # 5b. Validation set + (optional) InceptionV3 for VQ reconstruction eval
    # =========================================================================
    # Built independently of `accelerator.prepare()` so we keep the canonical
    # `DistributedSampler(shuffle=False, drop_last=False)` semantics: every
    # rank sees exactly `ceil(N / world_size)` samples (with a handful of
    # padded duplicates at the end), which is what `accelerator.gather` in
    # `run_vq_reconstruction_eval` relies on.
    val_loader = None
    inception = None
    if args.vq_eval_steps > 0:
        val_dataset = build_imagenet_val_dataset(
            args.vq_eval_data_dir, image_size=args.image_size,
        )
        if accelerator.use_distributed:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=False,
                drop_last=False,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=int(args.vq_eval_batch_size),
                sampler=val_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=int(args.vq_eval_batch_size),
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        if accelerator.is_main_process:
            log.info(
                f"VQ-eval val set: {len(val_dataset):,} images "
                f"({args.vq_eval_data_dir})"
            )

        if args.vq_eval_fid:
            # Heavy import + weight download: do it once, on every rank,
            # right before training starts (no per-eval cost). Inception
            # is small (~100MB) and runs on the same device.
            from tools.calculate_fid import InceptionV3  # noqa: E402

            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            inception = InceptionV3([block_idx]).to(device).eval()
            for p in inception.parameters():
                p.requires_grad = False
            if accelerator.is_main_process:
                log.info("VQ-eval: loaded InceptionV3 (FID enabled)")

    # =========================================================================
    # 6. Prepare
    # =========================================================================
    # Neither `ar` nor `vq` is wrapped yet (DDP / torch.compile run further
    # below in step 7), so no unwrap is needed here.
    update_ema(ema, ar, decay=0.0)
    update_ema(vq_ema, vq, decay=0.0)
    ar.eval(); ema.eval(); vq.eval(); vq_ema.eval()

    global_step = 0
    if args.resume_step > 0:
        path = f"{save_dir}/checkpoints/{args.resume_step:07d}.pt"
        ck = torch.load(path, map_location="cpu")
        ar.load_state_dict(ck["ar"]); ema.load_state_dict(ck["ema"])
        projectors.load_state_dict(ck["projectors"])
        # ``strict=False`` here: old ckpts (pre-codebook_used-removal) carry
        # a ``quantize.codebook_used`` key that no longer exists on the new
        # module, which would otherwise blow up resume.
        vq.load_state_dict(ck["vq"], strict=False)
        vq_loss_fn.discriminator.load_state_dict(ck["discriminator"])
        # Backwards compat: older ckpts (before the VQ-EMA patch) don't have
        # `vq_ema`. Fall back to the live VQ params -- the EMA will start
        # tracking from there with `args.vq_ema_decay`.
        if "vq_ema" in ck:
            vq_ema.load_state_dict(ck["vq_ema"], strict=False)
        else:
            vq_ema.load_state_dict(ck["vq"], strict=False)
            if accelerator.is_main_process:
                log.info("Resume ckpt has no `vq_ema`; initialised vq_ema from vq.")
        opt_ar.load_state_dict(ck["opt_ar"]); opt_vq.load_state_dict(ck["opt_vq"])
        opt_disc.load_state_dict(ck["opt_disc"]); global_step = ck["steps"]
        # LR schedulers: forward-compat with old ckpts (pre-warmup feature).
        # If absent, schedulers stay at last_epoch=0, ie warmup will replay
        # from scratch -- harmless and explicit. With the standard
        # ``--warmup-steps 0`` default, this branch is also a no-op (constant).
        if "sched_ar" in ck:
            sched_ar.load_state_dict(ck["sched_ar"])
        if "sched_vq" in ck:
            sched_vq.load_state_dict(ck["sched_vq"])
        if "sched_disc" in ck:
            sched_disc.load_state_dict(ck["sched_disc"])

    # ----- torch.compile -----------------------------------------------------
    # Mirrors REPA-E/train_repae.py:330-337. We compile the three "hot" modules
    # (vq, ar, discriminator-bearing loss). Done BEFORE accelerator.prepare so
    # DDP wraps the OptimizedModule -- the standard pattern.
    #
    # Notes specific to this codebase:
    #   * AR is called with two different kwargs combos every step
    #     (early_exit=True for VQ-step, False for AR-step) -> two compile
    #     variants; bumping cache size headroom matters.
    #   * EMA / checkpoint save / sampling deliberately reach through
    #     ``_orig_mod`` (see helper below) so they don't trigger extra
    #     recompilations or hold references to the OptimizedModule wrapper.
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512
    if args.compile:
        vq = torch.compile(vq, backend="inductor", mode="default")
        ar = torch.compile(ar, backend="inductor", mode="default")
        vq_loss_fn = torch.compile(vq_loss_fn, backend="inductor", mode="default")

    (ar, projectors, vq, vq_loss_fn,
     opt_ar, opt_vq, opt_disc,
     sched_ar, sched_vq, sched_disc,
     train_loader) = accelerator.prepare(
        ar, projectors, vq, vq_loss_fn,
        opt_ar, opt_vq, opt_disc,
        sched_ar, sched_vq, sched_disc,
        train_loader,
    )

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
        range(args.max_train_steps), initial=global_step, desc="Step",
        disable=not accelerator.is_local_main_process,
    )

    K = args.codebook_size

    # =========================================================================
    # 7. Training loop
    # =========================================================================
    while True:
        for raw_image, y in train_loader:
            raw_image = raw_image.to(device)        # uint8 [0, 255]
            labels = y.to(device)
            processed = preprocess_imgs_for_codec(raw_image)  # [-1, 1]
            B = processed.shape[0]
            L = block_size

            # ---- DINOv2 / target features (no grad) ---------------------------
            # ``--repa-encoder-layer`` selects which transformer block to align to:
            #   -1 (default) -> the standard ``forward_features`` output (post-norm)
            #    n >= 1      -> n-th block's pre-norm patch tokens
            # ``--repa-target-spnorm`` (and its alpha) optionally applies iREPA's
            # per-channel spatial normalisation BEFORE the cosine alignment.
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

            with accelerator.accumulate([vq, ar, projectors, vq_loss_fn]), accelerator.autocast():

                # =============================================================
                # ONE VQ forward per iteration. ``d`` is the raw squared
                # distance and is what controls AR-loss routing further down:
                # use it directly in step 1 to let REPA loss flow back to the
                # encoder, and use ``d.detach()`` in step 3 to fully cut the
                # AR step from the VQ branch (mirrors REPA-E's reuse of `z`
                # as `z.detach()` for the SiT step).
                # =============================================================
                # Always end-to-end: the VQ tokenizer is jointly trained with
                # the AR for the entire run (REPA-E style). One VQ forward per
                # iteration produces ``recon``/``d``/``indices`` reused below.
                # =============================================================
                vq.train()
                recon, z_q, (vq_l, commit_l, ent_l, usage), (_, _, indices), d = vq(
                    processed, return_distance=True,
                )

                # ----- Codebook decisiveness diagnostics ---------------------
                # Per-token natural softmax over codes (no temperature). Used
                # for cumulative top-k probability mass (how concentrated the
                # selection is) and for the equivalent uniform support size
                # ``exp(H)`` -- a "confused over N codes" reading of entropy.
                # All quantities below are per-token means -> linear over
                # batch -> ``gather().mean()`` recovers the global mean
                # exactly when local batch sizes are equal.
                with torch.no_grad():
                    d_det = d.detach().float()                         # (T, K)
                    K_codes = d_det.shape[-1]
                    k_max = min(1000, K_codes)

                    # `codebook/*` -- natural softmax over codes (T=1). Measures
                    # the codebook's INTRINSIC structure independent of how we
                    # sharpen the soft labels for AR.
                    log_probs_nat = F.log_softmax(-d_det, dim=-1)
                    probs_nat = log_probs_nat.exp()
                    nat_ent_local = -(probs_nat * log_probs_nat).sum(dim=-1).mean()
                    topk_probs, _ = torch.topk(probs_nat, k=k_max, dim=-1, sorted=True)
                    cum_top = torch.cumsum(topk_probs, dim=-1)
                    topk_local = {}
                    for k in (1, 10, 100, 1000):
                        idx = min(k, k_max) - 1
                        topk_local[k] = cum_top[:, idx].mean()

                    # `soft/*` -- same metrics but at `args.temperature`, i.e.
                    # the EXACT distribution fed into the AR forward via
                    # ``F.softmax(-d / T, dim=-1)`` below. Compare against the
                    # `codebook/*` versions to see how much the temperature is
                    # sharpening (or smoothing) the soft label:
                    #   soft/entropy < codebook/natural_entropy  -> T<1, sharper
                    #   soft/top1   > codebook/top1              -> closer to one-hot
                    soft_temp = float(args.temperature)
                    log_probs_soft = F.log_softmax(-d_det / soft_temp, dim=-1)
                    probs_soft = log_probs_soft.exp()
                    soft_ent_local = -(probs_soft * log_probs_soft).sum(dim=-1).mean()
                    soft_topk_probs, _ = torch.topk(probs_soft, k=k_max, dim=-1, sorted=True)
                    soft_cum_top = torch.cumsum(soft_topk_probs, dim=-1)
                    soft_topk_local = {}
                    for k in (1, 10, 100, 1000):
                        idx = min(k, k_max) - 1
                        soft_topk_local[k] = soft_cum_top[:, idx].mean()

                    # Global codebook usage from this step's indices.
                    # ``vq_model.VectorQuantizer`` keeps a per-rank sliding-
                    # window buffer of recent indices, so its ``usage`` is a
                    # local-only number that DOESN'T match a single-card
                    # stacked-batch run. Replace it with the union over all
                    # ranks of THIS step's indices.
                    indices_gathered = accelerator.gather(indices.detach())
                    codebook_usage_global = float(
                        torch.unique(indices_gathered).numel()
                    ) / float(K_codes)

                # =============================================================
                # 1) VQ generator step
                #    REPA loss path: soft = softmax(-d / T) (no detach) so the
                #    align loss flows back through soft -> d -> z -> encoder
                #    (and -> codebook embedding).
                # =============================================================
                ar.eval()
                requires_grad(ar, False); requires_grad(projectors, False)

                vq_total, vq_loss_dict = vq_loss_fn(
                    processed, recon,
                    quantizer_losses=(vq_l, commit_l, ent_l),
                    global_step=global_step,
                    mode="generator",
                )

                soft = F.softmax(-d / float(args.temperature), dim=-1).view(B, L, K)
                if args.vq_align_straight_through:
                    hard = F.one_hot(
                        indices.detach().view(B, L), num_classes=K,
                    ).to(dtype=soft.dtype, device=soft.device)
                    # Forward value is exactly one-hot (same as hard AR tokens),
                    # while backward follows the temperature-soft posterior.
                    soft = soft + (hard - soft).detach()
                # ``early_exit=True`` skips layers > encoder_depth + the
                # final ``norm`` and vocab projection. Gradients reaching the
                # encoder/codebook are byte-identical to a full forward (the
                # skipped sub-graph has no loss reference); the saving is
                # ~half the AR layers + a (dim x vocab_size=16384) matmul.
                _, hidden_at_tap = ar(
                    idx=soft[:, :-1, :], cond_idx=labels,
                    return_hidden_at=args.encoder_depth,
                    early_exit=True,
                )
                zs_tilde_vq = project_repa(hidden_at_tap, args.cls_token_num, projectors)
                proj_loss_vq = repa_align_loss(zs_tilde_vq, zs)
                vq_total = vq_total + args.vae_align_proj_coeff * proj_loss_vq

                accelerator.backward(vq_total)
                grad_norm_vq = None
                if accelerator.sync_gradients:
                    grad_norm_vq = accelerator.clip_grad_norm_(vq.parameters(), args.max_grad_norm)
                opt_vq.step(); sched_vq.step(); opt_vq.zero_grad(set_to_none=True)

                # =============================================================
                # 2) Discriminator step (uses the same ``recon``;
                #    ``vq_loss_fn`` internally calls ``recon.detach()``)
                # =============================================================
                d_loss, d_loss_dict = vq_loss_fn(
                    processed, recon,
                    quantizer_losses=(vq_l, commit_l, ent_l),
                    global_step=global_step,
                    mode="discriminator",
                )
                accelerator.backward(d_loss)
                grad_norm_disc = None
                if accelerator.sync_gradients:
                    grad_norm_disc = accelerator.clip_grad_norm_(vq_loss_fn.parameters(), args.max_grad_norm)
                opt_disc.step(); sched_disc.step(); opt_disc.zero_grad(set_to_none=True)

                # =============================================================
                # 3) AR step. Reuse ``d`` from the same forward as
                #    ``d.detach()`` -> a fresh leaf with the same values but no
                #    graph back to VQ. AR + projectors are the only params
                #    that get updated.
                # =============================================================
                ar.train()
                requires_grad(ar, True); requires_grad(projectors, True)

                # soft_ar = F.softmax(-d.detach() / float(args.temperature), dim=-1).view(B, L, K)
                ar_targets = indices.detach().view(B, L)

                logits, hidden_at_tap_ar = ar(
                    idx=ar_targets[:, :-1], cond_idx=labels,
                    return_hidden_at=args.encoder_depth,
                )
                ce_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    ar_targets.reshape(-1),
                )
                zs_tilde_ar = project_repa(hidden_at_tap_ar, args.cls_token_num, projectors)
                proj_loss_ar = repa_align_loss(zs_tilde_ar, zs)
                ar_total = ce_loss + args.proj_coeff * proj_loss_ar

                accelerator.backward(ar_total)
                grad_norm_ar = None
                if accelerator.sync_gradients:
                    grad_norm_ar = accelerator.clip_grad_norm_(
                        list(ar.parameters()) + list(projectors.parameters()),
                        args.max_grad_norm,
                    )
                opt_ar.step(); sched_ar.step(); opt_ar.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    # Same decay for AR EMA and VQ EMA so they lag the live
                    # params by the same number of steps -- otherwise the
                    # ``fid/ema`` eval (EMA AR + EMA VQ) loses its "time-
                    # aligned codebook layout" property and the FID number
                    # becomes a noisy mix of two different lag scales.
                    update_ema(
                        ema,
                        _orig(accelerator.unwrap_model(ar)),
                        decay=args.vq_ema_decay,
                    )
                    # VQ EMA: same cadence as AR EMA (only on grad-sync
                    # steps, ie when params actually moved). Update AFTER
                    # the AR step so we read live VQ params at their
                    # post-`opt_vq.step()` value, not in the middle of a
                    # gradient accumulation window.
                    update_ema(
                        vq_ema,
                        _orig(accelerator.unwrap_model(vq)),
                        decay=args.vq_ema_decay,
                    )

            # ---- logging --------------------------------------------------
            if accelerator.sync_gradients:
                progress.update(1); global_step += 1

                nat_ent_mean = accelerator.gather(nat_ent_local).mean().item()
                soft_ent_mean = accelerator.gather(soft_ent_local).mean().item()
                logs = {
                    "ar_total": accelerator.gather(ar_total).mean().item(),
                    "ce_loss": accelerator.gather(ce_loss).mean().item(),
                    "proj_loss_ar": accelerator.gather(proj_loss_ar).mean().item(),
                    # codebook decisiveness diagnostics (no temperature)
                    "codebook/natural_entropy_nats": nat_ent_mean,
                    "codebook/effective_size": math.exp(nat_ent_mean),
                    "codebook/top1_prob": accelerator.gather(topk_local[1]).mean().item(),
                    "codebook/top10_prob": accelerator.gather(topk_local[10]).mean().item(),
                    "codebook/top100_prob": accelerator.gather(topk_local[100]).mean().item(),
                    "codebook/top1000_prob": accelerator.gather(topk_local[1000]).mean().item(),
                    # soft/* uses args.temperature -- same set of metrics as
                    # codebook/* above but on the actual AR-input distribution.
                    "soft/entropy_nats": soft_ent_mean,
                    "soft/effective_size": math.exp(soft_ent_mean),
                    "soft/top1_prob": accelerator.gather(soft_topk_local[1]).mean().item(),
                    "soft/top10_prob": accelerator.gather(soft_topk_local[10]).mean().item(),
                    "soft/top100_prob": accelerator.gather(soft_topk_local[100]).mean().item(),
                    "soft/top1000_prob": accelerator.gather(soft_topk_local[1000]).mean().item(),
                    "codebook_usage": codebook_usage_global,
                    "temperature": args.temperature,
                    # lr/* -- effective LR after the scheduler step (single
                    # source of truth for warmup curves).
                    "lr/ar": opt_ar.param_groups[0]["lr"],
                }
                if grad_norm_ar is not None:
                    logs["grad_norm_ar"] = accelerator.gather(grad_norm_ar).mean().item()

                # VQ-/disc-specific metrics (the tokenizer is trained jointly
                # for the whole run, so these are always available).
                logs.update({
                    "vq_total": accelerator.gather(vq_total).mean().item(),
                    "proj_loss_vq": accelerator.gather(proj_loss_vq).mean().item(),
                    "reconstruction_loss": accelerator.gather(vq_loss_dict["reconstruction_loss"].mean()).mean().item(),
                    "perceptual_loss": accelerator.gather(vq_loss_dict["perceptual_loss"].mean()).mean().item(),
                    "vq_loss": accelerator.gather(vq_loss_dict["vq_loss"]).mean().item(),
                    "commit_loss": accelerator.gather(vq_loss_dict["commit_loss"]).mean().item(),
                    "entropy_loss": accelerator.gather(vq_loss_dict["entropy_loss"]).mean().item(),
                    "sample_entropy": accelerator.gather(vq_loss_dict["sample_entropy"]).mean().item(),
                    "avg_entropy": accelerator.gather(vq_loss_dict["avg_entropy"]).mean().item(),
                    # disc/* -- everything related to the discriminator /
                    # adversarial branch lives in its own namespace so the
                    # wandb panel doesn't get cluttered when GAN is off
                    # (discriminator_start) or weights are tuned.
                    "disc/weighted_gan_loss": accelerator.gather(vq_loss_dict["weighted_gan_loss"]).mean().item(),
                    "disc/gan_loss": accelerator.gather(vq_loss_dict["gan_loss"]).mean().item(),
                    "disc/discriminator_factor": accelerator.gather(vq_loss_dict["discriminator_factor"]).mean().item(),
                    "disc/d_loss": accelerator.gather(d_loss).mean().item(),
                    "disc/logits_real": accelerator.gather(d_loss_dict["logits_real"]).mean().item(),
                    "disc/logits_fake": accelerator.gather(d_loss_dict["logits_fake"]).mean().item(),
                    "lr/vq": opt_vq.param_groups[0]["lr"],
                    "lr/disc": opt_disc.param_groups[0]["lr"],
                })
                if grad_norm_vq is not None:
                    logs["grad_norm_vq"] = accelerator.gather(grad_norm_vq).mean().item()
                if grad_norm_disc is not None:
                    logs["disc/grad_norm"] = accelerator.gather(grad_norm_disc).mean().item()

                progress.set_postfix(**{k: f"{v:.3f}" if isinstance(v, float) else v for k, v in logs.items()})
                accelerator.log(logs, step=global_step)

            # ---- checkpoint ------------------------------------------------
            if accelerator.sync_gradients and (global_step % args.checkpointing_steps == 0 and global_step > 0):
                if accelerator.is_main_process:
                    ckpt = {
                        "ar": _orig(accelerator.unwrap_model(ar)).state_dict(),
                        "ema": ema.state_dict(),
                        "projectors": accelerator.unwrap_model(projectors).state_dict(),
                        "vq": _orig(accelerator.unwrap_model(vq)).state_dict(),
                        "vq_ema": vq_ema.state_dict(),
                        "discriminator": _orig(accelerator.unwrap_model(vq_loss_fn)).discriminator.state_dict(),
                        "opt_ar": opt_ar.state_dict(),
                        "opt_vq": opt_vq.state_dict(),
                        "opt_disc": opt_disc.state_dict(),
                        "sched_ar": sched_ar.state_dict(),
                        "sched_vq": sched_vq.state_dict(),
                        "sched_disc": sched_disc.state_dict(),
                        "args": vars(args),
                        "steps": global_step,
                    }
                    path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(ckpt, path); log.info(f"Saved checkpoint to {path}")

            # ---- recon + AR sample logging ---------------------------------
            # Visualisation only -- a handful of images for sanity checking.
            # We keep it cheap by running ENTIRELY on the main rank: no
            # gather, no other-rank participation. With 64 ranks gathering
            # would also flood the wandb panel.
            if accelerator.sync_gradients and (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                if accelerator.is_main_process and args.report_to == "wandb":
                    # Use the *original* (uncompiled) modules: ``generate()``
                    # repeatedly reshapes inputs and mutates KV caches, both
                    # of which would force OptimizedModule to recompile.
                    unwrapped_vq = _orig(accelerator.unwrap_model(vq))
                    unwrapped_ar = _orig(accelerator.unwrap_model(ar))
                    latent_size_vis = args.image_size // args.downsample_ratio
                    unwrapped_vq.eval(); unwrapped_ar.eval()

                    # (1) Reconstruction pairs (input | recon side-by-side).
                    n_recon = max(1, min(args.sampling_num, processed.shape[0]))
                    with torch.no_grad():
                        inp_log = processed[:n_recon]
                        recon_log, _ = unwrapped_vq(inp_log)
                        recon_log = (recon_log.clamp(-1, 1) + 1) / 2.0
                        inp_log = (inp_log.clamp(-1, 1) + 1) / 2.0

                    # (2) AR-generated samples (CFG, top-k/top-p, mirrors inference.py).
                    # NOTE: do NOT wrap ``generate`` in ``accelerator.autocast()``.
                    # ``setup_caches`` creates KVCache buffers with dtype taken
                    # from ``model.tok_embeddings.weight.dtype`` (fp32 under
                    # accelerate mixed-precision since params stay in fp32).
                    # With autocast on, attention activations would be bf16/fp16
                    # and the in-place ``k_cache[..., input_pos] = k_val`` write
                    # would raise a dtype mismatch. fp32 sampling runs only
                    # every ``--sampling-steps`` steps so perf cost is negligible.
                    n_sample = max(1, min(args.sampling_num, labels.shape[0]))
                    c_indices = labels[:n_sample].to(device=device, dtype=torch.long)
                    index_sample = generate(
                        unwrapped_ar, c_indices, latent_size_vis ** 2,
                        cfg_scale=args.sampling_cfg_scale,
                        cfg_interval=args.sampling_cfg_interval,
                        temperature=args.sampling_temperature,
                        top_k=args.sampling_top_k,
                        top_p=args.sampling_top_p,
                        sample_logits=True,
                    )
                    qzshape = [
                        c_indices.shape[0],
                        args.codebook_embed_dim,
                        latent_size_vis, latent_size_vis,
                    ]
                    with torch.no_grad():
                        sample_imgs = unwrapped_vq.decode_code(index_sample, qzshape)
                        sample_imgs = (sample_imgs.clamp(-1, 1) + 1) / 2.0

                    # Drop KV-cache state set up by ``generate`` so the next
                    # training-mode forward does not see stale caches.
                    unwrapped_ar.train()

                    accelerator.log(
                        {
                            "recon_grid": wandb.Image(
                                array2grid_pairs(
                                    inp_log.float(), recon_log.float(),
                                    pairs_per_row=args.sampling_pairs_per_row,
                                )
                            ),
                            "ar_samples": wandb.Image(array2grid(sample_imgs.float())),
                        },
                        step=global_step,
                    )

            # ---- FID eval --------------------------------------------------
            # Two FID numbers per trigger so the eval pairing stays *time-
            # aligned* (= the codebook layout the AR expects matches the
            # one the VQ decoder uses):
            #
            #   * ``fid/live`` -- live AR + live VQ (current params).
            #   * ``fid/ema``  -- EMA  AR + EMA  VQ (both lag the live params
            #                     by the same ~7000 steps with the shared
            #                     ``--vq-ema-decay`` (default 0.9999), so
            #                     they share a single "shadow" codebook
            #                     layout). This number is the more honest
            #                     proxy for inference quality.
            #
            # The old single "fid" key (EMA AR + live VQ) is gone -- it
            # crossed the EMA boundary between modules and could overstate
            # FID early in training while VQ was still drifting fast.
            # Sample dirs are step-tagged with `_live` / `_ema` so the two
            # runs don't clobber each other.
            do_eval = (
                args.eval_steps > 0
                and args.fid_reference_file
                and accelerator.sync_gradients
                and (global_step % args.eval_steps == 0 and global_step > 0)
            )
            if do_eval:
                unwrapped_ar_eval = _orig(accelerator.unwrap_model(ar))
                unwrapped_vq_eval = _orig(accelerator.unwrap_model(vq))

                prev_ar_training = unwrapped_ar_eval.training
                prev_vq_training = unwrapped_vq_eval.training
                unwrapped_ar_eval.eval()
                unwrapped_vq_eval.eval()

                fid_kwargs = dict(
                    accelerator=accelerator,
                    fid_num=args.eval_fid_num,
                    fid_reference_file=args.fid_reference_file,
                    num_classes=args.num_classes,
                    latent_size=latent_size,
                    codebook_embed_dim=args.codebook_embed_dim,
                    eval_per_proc_batch_size=args.eval_per_proc_batch_size,
                    sampling_temperature=args.eval_temperature,
                    sampling_top_k=args.eval_top_k,
                    sampling_top_p=args.eval_top_p,
                    sampling_cfg_scale=1.0,
                    sampling_cfg_interval=-1,
                    fid_batch_size=args.eval_fid_batch_size,
                    fid_num_workers=args.eval_fid_num_workers,
                    save_workers=args.eval_save_workers,
                    keep_samples=args.eval_keep_samples,
                    log=log if accelerator.is_main_process else None,
                )

                # ------ Run #1: live AR + live VQ -> fid_live -------------
                # No parameter swap needed -- the DDP/compile-wrapped ``ar``
                # already holds live params.
                eval_sample_dir_live = os.path.join(
                    args.output_dir, args.exp_name, "eval_samples",
                    f"{global_step:07d}_live",
                )
                fid_live = run_distributed_fid_eval(
                    ar_for_sampling=unwrapped_ar_eval,
                    vq_for_sampling=unwrapped_vq_eval,
                    sample_dir=eval_sample_dir_live,
                    **fid_kwargs,
                )

                # ------ Run #2: EMA AR + EMA VQ -> fid_ema ----------------
                # Stash live AR params (CPU clones to avoid GPU spike) and
                # swap in EMA params on the underlying nn.Module so the
                # same wrapped ``ar(...)`` call now uses EMA weights.
                # ``vq_ema`` is a separate, plain (un-wrapped), always-eval
                # nn.Module -- we just hand it to the helper, no swap.
                stashed = [p.detach().cpu().clone() for p in unwrapped_ar_eval.parameters()]
                for ema_p, mp in zip(ema.parameters(), unwrapped_ar_eval.parameters()):
                    mp.data.copy_(ema_p.to(mp.device).data)

                eval_sample_dir_ema = os.path.join(
                    args.output_dir, args.exp_name, "eval_samples",
                    f"{global_step:07d}_ema",
                )
                fid_ema = run_distributed_fid_eval(
                    ar_for_sampling=unwrapped_ar_eval,
                    vq_for_sampling=vq_ema,
                    sample_dir=eval_sample_dir_ema,
                    **fid_kwargs,
                )

                # Restore live AR params and prior train/eval states.
                # ``vq_ema`` was never touched (always eval, requires_grad=
                # False), so it needs no restore.
                for sp, mp in zip(stashed, unwrapped_ar_eval.parameters()):
                    mp.data.copy_(sp.to(mp.device).data)
                if prev_ar_training:
                    unwrapped_ar_eval.train()
                else:
                    unwrapped_ar_eval.eval()
                if prev_vq_training:
                    unwrapped_vq_eval.train()
                else:
                    unwrapped_vq_eval.eval()

                if accelerator.is_main_process:
                    # Namespace under `fid/` so the two curves group together
                    # in the wandb panel sidebar.
                    fid_logs = {}
                    if fid_live is not None:
                        fid_logs["fid/live"] = fid_live
                    if fid_ema is not None:
                        fid_logs["fid/ema"] = fid_ema
                    if fid_logs:
                        accelerator.log(fid_logs, step=global_step)

            # ---- VQ reconstruction eval (L1 / PSNR / SSIM / FID) ------------
            # Distinct from the AR-generation FID above: this measures how
            # well the *current* VQ reconstructs a held-out ImageNet val
            # split, using the same `preprocess_imgs_for_codec` path the
            # train loop uses. We run the eval TWICE per trigger: once on
            # the live VQ (-> wandb keys `vq_val/*`) and once on the EMA
            # VQ (-> `vq_val_ema/*`). The EMA is the smoother / more
            # downstream-friendly weight; comparing the two curves shows
            # how much the live params are still oscillating.
            #
            # `global_step == 1` runs once at the start as a sanity check:
            # for warm-started VQ (e.g. `vq_ds16_c2i.pt`) this gives a real
            # baseline PSNR/SSIM/FID number before training mutates anything,
            # AND it surfaces eval-path bugs (val loader, InceptionV3, gather
            # shapes, ...) within seconds rather than after the first
            # `--vq-eval-steps`-step interval. At step 1 the EMA equals the
            # live VQ (we did `decay=0.0` once at startup), so the two
            # metric sets will print identical numbers -- this is expected
            # and itself a sanity check.
            do_vq_eval = (
                val_loader is not None
                and args.vq_eval_steps > 0
                and accelerator.sync_gradients
                and (
                    global_step == 1
                    or (global_step % args.vq_eval_steps == 0 and global_step > 0)
                )
            )
            if do_vq_eval:
                unwrapped_vq_recon = _orig(accelerator.unwrap_model(vq))
                prev_vq_training = unwrapped_vq_recon.training

                # ---- live VQ ------------------------------------------
                unwrapped_vq_recon.eval()
                vq_metrics = run_vq_reconstruction_eval(
                    accelerator=accelerator,
                    vq=unwrapped_vq_recon,
                    val_loader=val_loader,
                    inception=inception,
                    eval_l1=args.vq_eval_l1,
                    eval_psnr=args.vq_eval_psnr,
                    eval_ssim=args.vq_eval_ssim,
                    log=log if accelerator.is_main_process else None,
                )
                if prev_vq_training:
                    unwrapped_vq_recon.train()

                # ---- EMA VQ -------------------------------------------
                # `vq_ema` is a plain (un-wrapped) nn.Module that is always
                # in eval mode, so no train/eval stash needed.
                vq_metrics_ema = run_vq_reconstruction_eval(
                    accelerator=accelerator,
                    vq=vq_ema,
                    val_loader=val_loader,
                    inception=inception,
                    eval_l1=args.vq_eval_l1,
                    eval_psnr=args.vq_eval_psnr,
                    eval_ssim=args.vq_eval_ssim,
                    log=log if accelerator.is_main_process else None,
                )
                vq_metrics_ema = {
                    k.replace("vq_val/", "vq_val_ema/"): v
                    for k, v in vq_metrics_ema.items()
                }

                if accelerator.is_main_process:
                    if vq_metrics:
                        accelerator.log(vq_metrics, step=global_step)
                    if vq_metrics_ema:
                        accelerator.log(vq_metrics_ema, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        log.info("Stage 1 done.")
    accelerator.end_training()


def parse_args():
    p = argparse.ArgumentParser()

    # logging
    p.add_argument("--output-dir", type=str, default="exps")
    p.add_argument("--exp-name", type=str, required=True)
    p.add_argument("--logging-dir", type=str, default="logs")
    p.add_argument("--report-to", type=str, default="wandb")
    p.add_argument("--wandb-project", type=str, default="src")
    p.add_argument("--sampling-steps", type=int, default=20000)
    # AR sampling controls (mirror inference.py / demo_sample_mode defaults)
    p.add_argument("--sampling-num", type=int, default=4,
                   help="Number of recon pairs / AR samples to log to wandb "
                        "at each sampling step. Computed on the main rank "
                        "only -- no gather across ranks.")
    p.add_argument("--sampling-pairs-per-row", type=int, default=4,
                   help="Pairs (input|recon) per row in the recon_grid wandb "
                        "image. Each pair occupies 2 cells, so the row width "
                        "is 2*N cells.")
    p.add_argument("--sampling-cfg-scale", type=float, default=4.0)
    p.add_argument("--sampling-cfg-interval", type=int, default=-1)
    p.add_argument("--sampling-temperature", type=float, default=1.0)
    p.add_argument("--sampling-top-k", type=int, default=2000)
    p.add_argument("--sampling-top-p", type=float, default=1.0)
    p.add_argument("--resume-step", type=int, default=0)
    p.add_argument("--cont-dir", type=str, default=None)

    # Online FID evaluation. Mirrors the offline `inference.py` pipeline but
    # runs in-process every `--eval-steps` global steps using the EMA model
    # and *without* CFG. Distributed: every rank generates a slice of the
    # samples (so 8 nodes x 8 GPUs makes 50k samples in a few seconds);
    # rank 0 alone calls `calculate_fid_given_paths`.
    p.add_argument("--eval-steps", type=int, default=0,
                   help="Run an EMA-based FID eval every N global steps. "
                        "Set <= 0 (default) to disable. Independent from "
                        "--sampling-steps which only logs a few wandb images.")
    p.add_argument("--eval-fid-num", type=int, default=50000,
                   help="Number of samples used for FID. Canonical value "
                        "is 50_000; 10_000 is a faster online proxy.")
    p.add_argument("--eval-per-proc-batch-size", type=int, default=32,
                   help="Per-rank batch size during FID sampling. Bigger = "
                        "faster but more peak VRAM during eval.")
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
                        "during FID sampling. Lets the next batch's GPU work "
                        "overlap with the previous batch's PIL save (~10-15%% "
                        "wall-time saving on AR-bottlenecked evals). Set 0 to "
                        "save synchronously (debug).")
    p.add_argument("--eval-keep-samples", action="store_true", default=False,
                   help="Keep generated PNGs after FID is computed (default: delete).")

    # VQ reconstruction eval (held-out ImageNet val: L1 / PSNR / SSIM / FID).
    # Independent of --eval-steps (which is for AR generation FID). No PNGs
    # are written -- we go straight from VQ recon to InceptionV3 features
    # in-memory, gather across ranks and compute the metrics.
    p.add_argument("--vq-eval-steps", type=int, default=0,
                   help="Run VQ reconstruction eval (L1/PSNR/SSIM/FID over "
                        "ImageNet val) every N global steps. Set <= 0 to "
                        "disable. Independent from --eval-steps.")
    p.add_argument("--vq-eval-data-dir", type=str, default=None,
                   help="ImageNet val folder (ImageFolder layout). Center-crop "
                        "+ short-side resize, no augmentation -- deterministic. "
                        "Required only when --vq-eval-steps > 0.")
    p.add_argument("--vq-eval-batch-size", type=int, default=32,
                   help="Per-rank batch size for VQ reconstruction eval.")
    p.add_argument("--vq-eval-l1", action=argparse.BooleanOptionalAction, default=True,
                   help="Toggle L1 metric in VQ recon eval.")
    p.add_argument("--vq-eval-psnr", action=argparse.BooleanOptionalAction, default=True,
                   help="Toggle PSNR metric in VQ recon eval.")
    p.add_argument("--vq-eval-ssim", action=argparse.BooleanOptionalAction, default=True,
                   help="Toggle SSIM metric (CPU + skimage; slowest).")
    p.add_argument("--vq-ema-decay", type=float, default=0.9999,
                   help="EMA decay applied to BOTH the AR shadow (`ema`) "
                        "and the VQ shadow (`vq_ema`). They are bound to "
                        "the same value so that the `fid/ema` eval "
                        "(EMA AR + EMA VQ) sees a time-aligned codebook "
                        "layout. Lower values track the live params more "
                        "aggressively; 0.9999 is conservative (~7k-step "
                        "half-life). (Flag kept as `vq-ema-decay` for "
                        "backwards compatibility with existing launch "
                        "scripts; it now controls AR EMA too.)")
    p.add_argument("--vq-eval-fid", action=argparse.BooleanOptionalAction, default=True,
                   help="Toggle FID metric. When True, loads InceptionV3 on "
                        "every rank at startup (~100MB).")

    # data
    p.add_argument("--data-dir", type=str, required=True,
                   help="Path to the ImageNet train folder following the canonical "
                        "<root>/<synset>/<file>.JPEG layout (consumed by "
                        "torchvision.datasets.ImageFolder).")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--random-hflip", action=argparse.BooleanOptionalAction, default=True,
                   help="Apply 50%% random horizontal flip in the data pipeline. "
                        "Standard for image generation training -- disable with "
                        "--no-random-hflip for ablation runs.")

    # VQ
    p.add_argument("--vq-model", type=str, default="VQ-16",
                   help="Tokenizer family / size. Looked up in "
                        "models.Tokenizers (currently: VQ-8 / VQ-16 / LFQ-16 / "
                        "IBQ-16). For LFQ-16 the codebook_embed_dim is forced "
                        "to log2(codebook_size) regardless of the CLI value; "
                        "for IBQ-16 use --codebook-embed-dim=256.")
    p.add_argument("--codebook-size", type=int, default=16384)
    p.add_argument("--codebook-embed-dim", type=int, default=8)
    p.add_argument("--downsample-ratio", type=int, default=16)
    p.add_argument("--vq-pretrained-ckpt", type=str, default=None)
    p.add_argument("--entropy-loss-ratio", type=float, default=0.05,
                   help="Override VectorQuantizer.entropy_loss_ratio.")
    p.add_argument(
        "--commit-loss-beta", type=float, default=None,
        help="Override the commit-MSE pre-multiplier inside the live "
             "quantizer (sets ``VectorQuantizer.beta`` for VQ, "
             "``LFQQuantizer.commit_loss_beta`` for LFQ, or "
             "``IBQQuantizer.beta`` for IBQ). Defaults to "
             "the dataclass value (0.25 for all). Set explicitly to "
             "override.",
    )

    # AR (LlamaGen)
    p.add_argument("--ar-model", type=str, default="LlamaGen-B",
                   choices=list(LlamaGen_models.keys()))
    p.add_argument("--cls-token-num", type=int, default=1)
    p.add_argument("--dropout-p", type=float, default=0.1)
    p.add_argument("--token-dropout-p", type=float, default=0.1)
    p.add_argument("--drop-path-rate", type=float, default=0.0)
    p.add_argument("--use-checkpoint", action="store_true", default=False)

    # Soft-quant
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Temperature T in softmax(-d / T). Default 1.0.")
    p.add_argument(
        "--vq-align-straight-through",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "In the VQ-side REPA alignment pass, feed AR a hard one-hot "
            "value but keep gradients from the temperature-soft posterior: "
            "soft + (one_hot(argmin(d)) - soft).detach(). This makes the AR "
            "forward value match hard tokens while still allowing the REPA "
            "loss to update VQ/codebook through softmax(-d / T)."
        ),
    )

    # REPA
    p.add_argument("--enc-type", type=str, default="dinov2-vit-b")
    p.add_argument("--encoder-depth", type=int, default=8,
                   help="1-based AR layer index whose hidden state is aligned with REPA target.")
    p.add_argument(
        "--repa-encoder-layer", type=int, default=-1,
        help=(
            "Which transformer block of the REPA encoder to align to. "
            "-1 (default) = the standard ``forward_features`` output, i.e. last "
            "block + final LayerNorm (what REPA has always used). 1..N = the "
            "n-th block's PRE-norm patch tokens. Per-encoder block counts: "
            "DINOv2/v3/SigLIP2/V-JEPA 2.1 ViT-B all have 12 blocks; ViT-L has 24."
        ),
    )
    p.add_argument(
        "--repa-target-spnorm", type=str, default="none",
        choices=["none", "demean", "zscore"],
        help=(
            "iREPA-style per-channel spatial normalisation on the REPA target "
            "before the cosine align loss. 'demean' = z - alpha * mean_l z "
            "(kill DC only); 'zscore' = also divide by per-channel spatial std "
            "(iREPA's actual setting). Default: none (vanilla REPA)."
        ),
    )
    p.add_argument(
        "--repa-spnorm-alpha", type=float, default=0.6,
        help=(
            "Mean-subtraction strength for --repa-target-spnorm. iREPA's LDM "
            "uses 0.6 (default); JiT uses 0.8; alpha=1.0 makes the per-channel "
            "spatial mean exactly zero."
        ),
    )
    p.add_argument("--projector-dim", type=int, default=2048)
    p.add_argument("--proj-coeff", type=float, default=0.5,
                   help="Coefficient for the AR-side REPA align loss.")
    p.add_argument("--vae-align-proj-coeff", type=float, default=1.5,
                   help="Coefficient for the VQ-side REPA align loss.")

    # Loss / GAN
    p.add_argument("--loss-cfg-path", type=str, default="src/configs/loss_l1_lpips_gan_vq.yaml")
    p.add_argument("--disc-pretrained-ckpt", type=str, default=None)

    # Stage-0 warm start. Single flag that loads `vq` + `vq_ema` +
    # `discriminator` from a `train_tokenizer.py` checkpoint (whose layout is a
    # subset of the Stage-1 ckpt format). When set, overrides
    # `--vq-pretrained-ckpt` and `--disc-pretrained-ckpt`. Optimizer /
    # scheduler state is intentionally NOT migrated -- Stage 1 has a
    # different loss recipe so reusing Adam moments is more likely to hurt.
    p.add_argument("--stage0-init-ckpt", type=str, default=None,
                   help="Path to a `train_tokenizer.py` checkpoint. Loads "
                        "vq + vq_ema + discriminator from a single file. "
                        "Overrides --vq-pretrained-ckpt / --disc-pretrained-ckpt.")

    # Optim
    p.add_argument("--ar-lr", type=float, default=1e-4)
    p.add_argument("--vq-lr", type=float, default=1e-4)
    p.add_argument("--disc-lr", type=float, default=1e-4)
    p.add_argument(
        "--warmup-steps", type=int, default=1,
        help=(
            "Linear warmup -> constant LR for AR / VQ / disc. 0 = no warmup "
            "(lr=base from step 0). 1 = step-0 lr is 0 (no parameter update); "
            "useful so eval at global_step==1 reports the *pretrained* "
            "baseline. >1 = ramp linearly across N optimizer steps."
        ),
    )
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--adam-weight-decay", type=float, default=0.0)
    p.add_argument("--adam-epsilon", type=float, default=1e-8)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--max-train-steps", type=int, default=400000)
    p.add_argument("--checkpointing-steps", type=int, default=50000)

    # Misc
    p.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="Apply torch.compile (inductor) to vq / ar / vq_loss_fn. "
                        "EMA, ckpt save and AR sampling reach through "
                        "_orig_mod, so behaviour is identical with --no-compile.")
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
