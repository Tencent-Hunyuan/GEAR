"""Stage 0 of the src pipeline: train the VQ tokenizer alone.

Stage 0 strips out everything AR / REPA / DINOv2 from `train_gear.py`
and keeps just the standard VAE-GAN recipe used for VQ tokenizer pre-
training. Per training step there are **two** optimizer updates:

1. **VQ generator step** (`opt_vq`):
   reconstruction loss + perceptual loss (LPIPS) + GAN gen loss
   (-mean(D(recon))) + VQ losses (vq + commit + entropy).

2. **Discriminator step** (`opt_disc`):
   hinge loss on real/fake (gen frozen).

The checkpoint format is a *strict subset* of the Stage-1 checkpoint
(same key names, just no AR / projector / AR-EMA entries). This means
``train_gear.py`` can warm-start the VQ + discriminator + VQ-EMA from a
Stage-0 ckpt with a single ``--stage0-init-ckpt`` flag.
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

from src.dataset import (  # noqa: E402
    build_imagenet_dataset,
    build_imagenet_val_dataset,
)
from src.losses import ReconstructionLossVQ  # noqa: E402
from src.utils import (  # noqa: E402
    count_trainable_params,
    get_constant_schedule_with_warmup,
    load_pretrained_tokenizer_state_dict,
    preprocess_imgs_for_codec,
    run_vq_reconstruction_eval,
)


logger = get_logger(__name__)


# =============================================================================
# Helpers (kept identical to train_gear.py so the two files stay in sync)
# =============================================================================
def array2grid(x, nrow=None):
    if nrow is None:
        nrow = round(math.sqrt(x.size(0)))
    g = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    g = g.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return g


def array2grid_pairs(inputs, recons, pairs_per_row: int = 4):
    """Lay out (input, recon) pairs side-by-side in a grid. See train_gear.py."""
    assert inputs.shape == recons.shape, (
        f"inputs.shape {tuple(inputs.shape)} != recons.shape {tuple(recons.shape)}"
    )
    pairs = torch.stack([inputs, recons], dim=1).reshape(-1, *inputs.shape[1:])
    return array2grid(pairs, nrow=2 * pairs_per_row)


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

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, args.exp_name)
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
    vq_kwargs = dict(
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    )
    # Optional grouped-entropy override (only consumed by the GLFQ family,
    # e.g. ``--vq-model=GLFQ-16``). Forwarded *only when explicitly set* so
    # VQ / LFQ / IBQ factories -- which don't accept ``num_codebooks`` -- are
    # entirely unaffected and keep their existing behaviour.
    if args.entropy_num_groups is not None:
        vq_kwargs["num_codebooks"] = int(args.entropy_num_groups)
    vq = VQ_models[args.vq_model](**vq_kwargs)
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
        # Both VQ and LFQ pre-multiply the commit MSE inside the quantizer
        # (so the outer loss only multiplies by ``quantizer_weight``).
        # ``VectorQuantizer`` calls the field ``beta``; ``LFQQuantizer``
        # calls it ``commit_loss_beta`` (named for clarity given LFQ has
        # no learnable codebook). Override whichever the live quantizer
        # exposes so a single CLI flag works for either tokenizer family.
        if hasattr(vq.quantize, "commit_loss_beta"):
            vq.quantize.commit_loss_beta = float(args.commit_loss_beta)
        elif hasattr(vq.quantize, "beta"):
            vq.quantize.beta = float(args.commit_loss_beta)
    vq = vq.to(device)

    assert args.image_size % args.downsample_ratio == 0, \
        "Image size must be divisible by VQ downsample ratio."

    # VQ EMA: a frozen, exponentially-averaged copy of the live VQ. Same role
    # as in Stage 1 (smooth weight for downstream eval / FID). Built BEFORE
    # accelerator.prepare so it stays a plain local nn.Module (not DDP-
    # wrapped) and is updated by hand after every grad-sync VQ optimizer
    # step. requires_grad=False everywhere -- pure storage.
    vq_ema = copy.deepcopy(vq).to(device)
    requires_grad(vq_ema, False)
    vq_ema.eval()

    # =========================================================================
    # 2. Loss + discriminator
    # =========================================================================
    loss_cfg = OmegaConf.load(args.loss_cfg_path)
    vq_loss_fn = ReconstructionLossVQ(loss_cfg).to(device)
    if args.disc_pretrained_ckpt:
        disc_state = torch.load(args.disc_pretrained_ckpt, map_location=device)
        vq_loss_fn.discriminator.load_state_dict(disc_state)
        if accelerator.is_main_process:
            log.info(f"Loaded discriminator init from {args.disc_pretrained_ckpt}")

    # ``NLayerDiscriminator`` uses plain ``nn.BatchNorm2d``; without sync the
    # per-rank running stats diverge across ranks. Mirror Stage 1: convert
    # the discriminator's BNs to SyncBatchNorm so all ranks see the same
    # global statistics. Must happen *before* prepare so the converted
    # module gets wrapped by DDP.
    if accelerator.use_distributed:
        vq_loss_fn.discriminator = (
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(vq_loss_fn.discriminator)
        )

    if accelerator.is_main_process:
        log.info(f"VQ params       : {sum(p.numel() for p in vq.parameters()):,}")
        log.info(f"Trainable VQ    : {count_trainable_params(vq):,}")
        log.info(f"Discriminator   : {sum(p.numel() for p in vq_loss_fn.discriminator.parameters()):,}")

    # =========================================================================
    # 3. Optimizers
    # =========================================================================
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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

    sched_vq = get_constant_schedule_with_warmup(opt_vq, num_warmup_steps=args.warmup_steps)
    sched_disc = get_constant_schedule_with_warmup(opt_disc, num_warmup_steps=args.warmup_steps)

    # =========================================================================
    # 4. Data
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
    # 4b. Validation set + (optional) InceptionV3 for VQ reconstruction eval
    # =========================================================================
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
            from tools.calculate_fid import InceptionV3  # noqa: E402

            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            inception = InceptionV3([block_idx]).to(device).eval()
            for p in inception.parameters():
                p.requires_grad = False
            if accelerator.is_main_process:
                log.info("VQ-eval: loaded InceptionV3 (FID enabled)")

    # =========================================================================
    # 5. Prepare
    # =========================================================================
    # Initialise vq_ema as an exact copy of the live vq (decay=0.0). Done
    # AFTER any pretrained loading and BEFORE accelerator.prepare so that
    # vq_ema stays a plain (un-wrapped) nn.Module.
    update_ema(vq_ema, vq, decay=0.0)
    vq.eval(); vq_ema.eval()

    global_step = 0
    if args.resume_step > 0:
        ckpt_path = f"{args.cont_dir}/checkpoints/{args.resume_step:07d}.pt"
        ck = torch.load(ckpt_path, map_location="cpu")
        # ``strict=False`` here: old ckpts (pre-codebook_used-removal) carry
        # a ``quantize.codebook_used`` key that no longer exists on the new
        # module, which would otherwise blow up resume.
        vq.load_state_dict(ck["vq"], strict=False)
        vq_loss_fn.discriminator.load_state_dict(ck["discriminator"])
        if "vq_ema" in ck:
            vq_ema.load_state_dict(ck["vq_ema"], strict=False)
        else:
            vq_ema.load_state_dict(ck["vq"], strict=False)
            if accelerator.is_main_process:
                log.info("Resume ckpt has no `vq_ema`; initialised vq_ema from vq.")
        opt_vq.load_state_dict(ck["opt_vq"])
        opt_disc.load_state_dict(ck["opt_disc"])
        global_step = ck["steps"]
        if "sched_vq" in ck:
            sched_vq.load_state_dict(ck["sched_vq"])
        if "sched_disc" in ck:
            sched_disc.load_state_dict(ck["sched_disc"])

    # ----- torch.compile -----------------------------------------------------
    # Same pattern as Stage 1: compile the two "hot" modules (vq, vq_loss_fn)
    # before accelerator.prepare so DDP wraps the OptimizedModule.
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512
    if args.compile:
        vq = torch.compile(vq, backend="inductor", mode="default")
        vq_loss_fn = torch.compile(vq_loss_fn, backend="inductor", mode="default")

    (vq, vq_loss_fn,
     opt_vq, opt_disc,
     sched_vq, sched_disc,
     train_loader) = accelerator.prepare(
        vq, vq_loss_fn,
        opt_vq, opt_disc,
        sched_vq, sched_disc,
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

    # =========================================================================
    # 6. Training loop
    # =========================================================================
    while True:
        for raw_image, _y in train_loader:
            raw_image = raw_image.to(device)        # uint8 [0, 255]
            processed = preprocess_imgs_for_codec(raw_image)  # [-1, 1]

            with accelerator.accumulate([vq, vq_loss_fn]), accelerator.autocast():

                # =============================================================
                # ONE VQ forward per iteration. ``return_distance=True`` is
                # only used here for the codebook decisiveness diagnostics
                # (`codebook/*`) -- the training losses themselves do not
                # need ``d``.
                # =============================================================
                vq.train()
                recon, z_q, (vq_l, commit_l, ent_l, usage), (_, _, indices), d = vq(
                    processed, return_distance=True,
                )

                # ----- Codebook decisiveness diagnostics ---------------------
                # Same `codebook/*` block as Stage 1 (no `soft/*` since
                # there's no temperature in Stage 0).
                with torch.no_grad():
                    d_det = d.detach().float()                         # (T, K)
                    K_codes = d_det.shape[-1]
                    k_max = min(1000, K_codes)

                    log_probs_nat = F.log_softmax(-d_det, dim=-1)
                    probs_nat = log_probs_nat.exp()
                    nat_ent_local = -(probs_nat * log_probs_nat).sum(dim=-1).mean()
                    topk_probs, _ = torch.topk(probs_nat, k=k_max, dim=-1, sorted=True)
                    cum_top = torch.cumsum(topk_probs, dim=-1)
                    topk_local = {}
                    for k in (1, 10, 100, 1000):
                        idx = min(k, k_max) - 1
                        topk_local[k] = cum_top[:, idx].mean()

                    # Global codebook usage from this step's indices.
                    indices_gathered = accelerator.gather(indices.detach())
                    codebook_usage_global = float(
                        torch.unique(indices_gathered).numel()
                    ) / float(K_codes)

                # =============================================================
                # 1) VQ generator step
                # =============================================================
                vq_total, vq_loss_dict = vq_loss_fn(
                    processed, recon,
                    quantizer_losses=(vq_l, commit_l, ent_l),
                    global_step=global_step,
                    mode="generator",
                )

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

                if accelerator.sync_gradients:
                    # VQ EMA update on grad-sync steps only (i.e. when params
                    # actually moved). Done AFTER the disc step so we read
                    # post-`opt_vq.step()` live VQ params.
                    update_ema(
                        vq_ema,
                        _orig(accelerator.unwrap_model(vq)),
                        decay=args.vq_ema_decay,
                    )

            # ---- logging --------------------------------------------------
            if accelerator.sync_gradients:
                progress.update(1); global_step += 1

                nat_ent_mean = accelerator.gather(nat_ent_local).mean().item()
                logs = {
                    "vq_total": accelerator.gather(vq_total).mean().item(),
                    "reconstruction_loss": accelerator.gather(vq_loss_dict["reconstruction_loss"].mean()).mean().item(),
                    "perceptual_loss": accelerator.gather(vq_loss_dict["perceptual_loss"].mean()).mean().item(),
                    "vq_loss": accelerator.gather(vq_loss_dict["vq_loss"]).mean().item(),
                    "commit_loss": accelerator.gather(vq_loss_dict["commit_loss"]).mean().item(),
                    "entropy_loss": accelerator.gather(vq_loss_dict["entropy_loss"]).mean().item(),
                    "sample_entropy": accelerator.gather(vq_loss_dict["sample_entropy"]).mean().item(),
                    "avg_entropy": accelerator.gather(vq_loss_dict["avg_entropy"]).mean().item(),
                    # codebook decisiveness diagnostics (no temperature)
                    "codebook/natural_entropy_nats": nat_ent_mean,
                    "codebook/effective_size": math.exp(nat_ent_mean),
                    "codebook/top1_prob": accelerator.gather(topk_local[1]).mean().item(),
                    "codebook/top10_prob": accelerator.gather(topk_local[10]).mean().item(),
                    "codebook/top100_prob": accelerator.gather(topk_local[100]).mean().item(),
                    "codebook/top1000_prob": accelerator.gather(topk_local[1000]).mean().item(),
                    # disc/* -- everything related to the discriminator /
                    # adversarial branch lives in its own namespace.
                    "disc/weighted_gan_loss": accelerator.gather(vq_loss_dict["weighted_gan_loss"]).mean().item(),
                    "disc/gan_loss": accelerator.gather(vq_loss_dict["gan_loss"]).mean().item(),
                    "disc/discriminator_factor": accelerator.gather(vq_loss_dict["discriminator_factor"]).mean().item(),
                    "disc/d_loss": accelerator.gather(d_loss).mean().item(),
                    "disc/logits_real": accelerator.gather(d_loss_dict["logits_real"]).mean().item(),
                    "disc/logits_fake": accelerator.gather(d_loss_dict["logits_fake"]).mean().item(),
                    "codebook_usage": codebook_usage_global,
                    # lr/* -- effective LR after the scheduler step.
                    "lr/vq": opt_vq.param_groups[0]["lr"],
                    "lr/disc": opt_disc.param_groups[0]["lr"],
                }
                if grad_norm_vq is not None:
                    logs["grad_norm_vq"] = accelerator.gather(grad_norm_vq).mean().item()
                if grad_norm_disc is not None:
                    logs["disc/grad_norm"] = accelerator.gather(grad_norm_disc).mean().item()

                progress.set_postfix(**{k: f"{v:.3f}" if isinstance(v, float) else v for k, v in logs.items()})
                accelerator.log(logs, step=global_step)

            # ---- checkpoint ------------------------------------------------
            # Strict subset of the Stage-1 ckpt format (no `ar`/`ema`/
            # `projectors`/`opt_ar`/`sched_ar`). `train_gear.py
            # --stage0-init-ckpt` consumes this directly.
            if accelerator.sync_gradients and (global_step % args.checkpointing_steps == 0 and global_step > 0):
                if accelerator.is_main_process:
                    ckpt = {
                        "vq": _orig(accelerator.unwrap_model(vq)).state_dict(),
                        "vq_ema": vq_ema.state_dict(),
                        "discriminator": _orig(accelerator.unwrap_model(vq_loss_fn)).discriminator.state_dict(),
                        "opt_vq": opt_vq.state_dict(),
                        "opt_disc": opt_disc.state_dict(),
                        "sched_vq": sched_vq.state_dict(),
                        "sched_disc": sched_disc.state_dict(),
                        "args": vars(args),
                        "steps": global_step,
                        "stage": 0,
                    }
                    path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(ckpt, path); log.info(f"Saved checkpoint to {path}")

            # ---- recon visualisation --------------------------------------
            # Stage 0 has no AR sampling, so we only log the (input | recon)
            # pair grid. Cheap, main-rank only.
            if accelerator.sync_gradients and (
                (args.eval_at_step_1 and global_step == 1)
                or (global_step % args.sampling_steps == 0 and global_step > 0)
            ):
                if accelerator.is_main_process and args.report_to == "wandb":
                    unwrapped_vq = _orig(accelerator.unwrap_model(vq))
                    unwrapped_vq.eval()

                    n_recon = max(1, min(args.sampling_num, processed.shape[0]))
                    with torch.no_grad():
                        inp_log = processed[:n_recon]
                        recon_log, _ = unwrapped_vq(inp_log)
                        recon_log = (recon_log.clamp(-1, 1) + 1) / 2.0
                        inp_log = (inp_log.clamp(-1, 1) + 1) / 2.0

                    accelerator.log(
                        {
                            "recon_grid": wandb.Image(
                                array2grid_pairs(
                                    inp_log.float(), recon_log.float(),
                                    pairs_per_row=args.sampling_pairs_per_row,
                                )
                            ),
                        },
                        step=global_step,
                    )

            # ---- VQ reconstruction eval (L1 / PSNR / SSIM / FID) ------------
            # Same recipe as Stage 1: run the eval TWICE (live VQ -> wandb
            # `vq_val/*`, EMA VQ -> `vq_val_ema/*`) so we can monitor both
            # the noisy live params and the smoother EMA shadow.
            do_vq_eval = (
                val_loader is not None
                and args.vq_eval_steps > 0
                and accelerator.sync_gradients
                and (
                    (args.eval_at_step_1 and global_step == 1)
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
        log.info("Stage 0 done.")
    accelerator.end_training()


def parse_args():
    p = argparse.ArgumentParser()

    # logging
    p.add_argument("--output-dir", type=str, default="exps")
    p.add_argument("--exp-name", type=str, required=True)
    p.add_argument("--logging-dir", type=str, default="logs")
    p.add_argument("--report-to", type=str, default="wandb")
    p.add_argument("--wandb-project", type=str, default="src")
    p.add_argument("--sampling-steps", type=int, default=10000)
    p.add_argument("--sampling-num", type=int, default=4,
                   help="Number of recon pairs to log to wandb at each "
                        "sampling step. Computed on the main rank only.")
    p.add_argument("--sampling-pairs-per-row", type=int, default=4,
                   help="Pairs (input|recon) per row in the recon_grid wandb image.")
    p.add_argument("--resume-step", type=int, default=0)
    p.add_argument("--cont-dir", type=str, default=None)

    # VQ reconstruction eval (held-out ImageNet val: L1 / PSNR / SSIM / FID).
    p.add_argument("--vq-eval-steps", type=int, default=0,
                   help="Run VQ reconstruction eval every N global steps. "
                        "Set <= 0 to disable.")
    p.add_argument("--vq-eval-data-dir", type=str, default=None,
                   help="ImageNet val folder (ImageFolder layout). Required "
                        "only when --vq-eval-steps > 0.")
    p.add_argument("--vq-eval-batch-size", type=int, default=32,
                   help="Per-rank batch size for VQ reconstruction eval.")
    p.add_argument("--vq-eval-l1", action=argparse.BooleanOptionalAction, default=True,
                   help="Toggle L1 metric in VQ recon eval.")
    p.add_argument("--vq-eval-psnr", action=argparse.BooleanOptionalAction, default=True,
                   help="Toggle PSNR metric in VQ recon eval.")
    p.add_argument("--vq-eval-ssim", action=argparse.BooleanOptionalAction, default=True,
                   help="Toggle SSIM metric (CPU + skimage; slowest).")
    p.add_argument("--vq-ema-decay", type=float, default=0.9999,
                   help="EMA decay for the VQ shadow weights.")
    p.add_argument("--eval-at-step-1", action=argparse.BooleanOptionalAction, default=True,
                   help="Run the recon-grid + VQ reconstruction eval once at "
                        "global_step==1. Useful for warm-started runs (captures "
                        "the pretrained baseline). Pass --no-eval-at-step-1 for "
                        "from-scratch runs where the step-1 numbers are just the "
                        "random init.")
    p.add_argument("--vq-eval-fid", action=argparse.BooleanOptionalAction, default=True,
                   help="Toggle FID metric. When True, loads InceptionV3 on every rank at startup.")

    # data
    p.add_argument("--data-dir", type=str, required=True,
                   help="Path to the ImageNet train folder following the canonical "
                        "<root>/<synset>/<file>.JPEG layout (consumed by "
                        "torchvision.datasets.ImageFolder).")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000,
                   help="Kept for ImageFolder compat; Stage 0 ignores labels.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--random-hflip", action=argparse.BooleanOptionalAction, default=True,
                   help="Apply 50%% random horizontal flip in the data pipeline.")

    # VQ
    p.add_argument("--vq-model", type=str, default="VQ-16",
                   help="Tokenizer family / size. Looked up in "
                        "models.Tokenizers (currently: VQ-8 / VQ-16 / LFQ-16 / "
                        "IBQ-16). For LFQ-16 the codebook_embed_dim is forced "
                        "to log2(codebook_size) regardless of the CLI value; "
                        "for IBQ-16 use --codebook-embed-dim=256 (matches the "
                        "TencentARC pretrain).")
    p.add_argument("--codebook-size", type=int, default=16384)
    p.add_argument("--codebook-embed-dim", type=int, default=8)
    p.add_argument("--downsample-ratio", type=int, default=16)
    p.add_argument("--vq-pretrained-ckpt", type=str, default=None)
    p.add_argument("--entropy-loss-ratio", type=float, default=0.05,
                   help="Override VectorQuantizer.entropy_loss_ratio.")
    p.add_argument("--entropy-num-groups", type=int, default=None,
                   help="Only for the GLFQ family (e.g. --vq-model=GLFQ-16). "
                        "Number of groups the dim is split into for the "
                        "(grouped) entropy aux loss; the per-group sub-codebook "
                        "is 2**(dim/groups). The quantization itself stays "
                        "per-dimension {-1,+1} regardless. Leave unset for "
                        "VQ / LFQ / IBQ (they don't accept this flag).")
    p.add_argument(
        "--commit-loss-beta", type=float, default=None,
        help="Override the commit-MSE pre-multiplier inside the live "
             "quantizer (sets ``VectorQuantizer.beta`` for VQ, "
             "``LFQQuantizer.commit_loss_beta`` for LFQ, or "
             "``IBQQuantizer.beta`` for IBQ). Defaults to "
             "the dataclass value (0.25 for all, matching LlamaGen's "
             "VQ-VAE recipe, MAGVIT2's pretrain_lfqgan_256_16384.yaml and "
             "IBQ's pretrain_ibqgan_16384.yaml ``beta: 0.25``). Set "
             "explicitly to override.",
    )

    # Loss / GAN
    p.add_argument("--loss-cfg-path", type=str, default="src/configs/loss_l1_lpips_gan_vq.yaml")
    p.add_argument("--disc-pretrained-ckpt", type=str, default=None)

    # Optim
    p.add_argument("--vq-lr", type=float, default=1e-4)
    p.add_argument("--disc-lr", type=float, default=1e-4)
    p.add_argument(
        "--warmup-steps", type=int, default=1,
        help=(
            "Linear warmup -> constant LR for VQ / disc. 0 = no warmup. "
            "1 = step-0 lr is 0 (no parameter update); useful so eval at "
            "global_step==1 reports the *pretrained* baseline."
        ),
    )
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--adam-weight-decay", type=float, default=0.0)
    p.add_argument("--adam-epsilon", type=float, default=1e-8)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--max-train-steps", type=int, default=400000)
    p.add_argument("--checkpointing-steps", type=int, default=25000)

    # Misc
    p.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="Apply torch.compile to vq / vq_loss_fn.")
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
