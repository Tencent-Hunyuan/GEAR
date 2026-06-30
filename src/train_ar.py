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
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datetime import timedelta
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
from models.llamagen import LlamaGen_models  # noqa: E402
from models.generate import generate  # noqa: E402  - AR sampling

from src.dataset import build_imagenet_dataset  # noqa: E402
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


# =============================================================================
# Main
# =============================================================================
def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    # 10-min NCCL collective timeout: turns silent hangs into a real
    # exception with a Python traceback. Default is 30 min which is too
    # long for our debugging loop; combined with the env vars in run.sh
    # (TORCH_NCCL_BLOCKING_WAIT=1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600)
    # this guarantees we get a crash within ~10 min of any rank skew.
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=10))
    # IMPORTANT: ``rng_types=[]`` disables accelerate's per-epoch
    # ``broadcast(rng_state, src=0)`` inside ``DataLoaderShard.__iter__``.
    # On 64 ranks that broadcast was the root cause of the silent hang
    # at ~step 5000 (= end of first epoch on ImageNet @ bs=256): if even
    # ONE rank entered the new epoch's ``__iter__`` ~1 collective ahead of
    # the others (which happens naturally from accumulated NCCL skew), it
    # would block in broadcast while the other 63 ranks were still inside
    # the previous step's allreduce -> classic collective-ordering deadlock.
    # We don't need accelerate's RNG sync because we drive the shuffle
    # ourselves via ``DistributedSampler(..., seed=args.seed)`` +
    # ``set_epoch(epoch)`` below, which deterministically produces the same
    # global permutation on all ranks without any runtime broadcast.
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
    # loaded: the EMA shadow (``vq_ema``, DEFAULT) or the live params
    # (``vq``, via --no-vq-use-ema). EMA is what Stage 1 itself reports
    # under ``vq_val_ema/*`` and is the recommended choice for the frozen
    # tokenizer in Stage 2 / inference; the live copy is kept around for
    # ablations and continued training. Has no effect on legacy / MAGVIT2
    # ckpts (those expose only one tensor source).
    #
    # NOTE: this is the *base* load from --vq-ckpt. If --ar-init-ckpt is a
    # Stage-1 ckpt that bundles its own tokenizer, the VQ is re-loaded from
    # that ckpt further down (see --vq-override-from-ar-init) so the AR and
    # its tokenizer stay in lockstep.
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
    # 2. AR + projectors
    # =========================================================================
    assert args.image_size % args.downsample_ratio == 0
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
        if projectors is not None and "projectors" in ar_ck:
            projectors.load_state_dict(ar_ck["projectors"])

        # --- Override the frozen VQ with the tokenizer bundled in the SAME
        # Stage-1 ckpt --------------------------------------------------------
        # A Stage-1 ckpt (train_gear.py) saves the AR *and* its co-trained
        # tokenizer (``vq`` live + ``vq_ema`` shadow). When we warm-start the
        # AR from such a ckpt, the only tokenizer that's guaranteed to match
        # the AR's training-time token statistics is THAT ckpt's VQ -- not
        # whatever ``--vq-ckpt`` happens to point at. So, by default, if the
        # ar-init ckpt carries VQ weights we re-load the VQ from it (this
        # overrides the earlier load from ``--vq-ckpt``). ``--vq-use-ema``
        # selects the EMA shadow (default) vs the live params, with a silent
        # fallback to whichever is present. Pass ``--no-vq-override-from-ar-init``
        # to keep the ``--vq-ckpt`` tokenizer instead.
        if args.vq_override_from_ar_init:
            vq_key = None
            if args.vq_use_ema and isinstance(ar_ck.get("vq_ema"), dict):
                vq_key = "vq_ema"
            elif isinstance(ar_ck.get("vq"), dict):
                vq_key = "vq"
            elif isinstance(ar_ck.get("vq_ema"), dict):
                vq_key = "vq_ema"
            if vq_key is not None:
                vq_sd = {
                    (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
                    for k, v in ar_ck[vq_key].items()
                }
                vmiss, vunexp = vq.load_state_dict(vq_sd, strict=False)
                vq.eval()
                for p in vq.parameters():
                    p.requires_grad = False
                if accelerator.is_main_process:
                    log.info(
                        f"Overrode VQ with ar-init-ckpt['{vq_key}'] "
                        f"(use_ema={args.vq_use_ema}) so the frozen tokenizer "
                        f"matches the AR's Stage-1 training tokenizer | "
                        f"missing={len(vmiss)}, unexpected={len(vunexp)}"
                    )
            elif accelerator.is_main_process:
                log.info(
                    f"--vq-override-from-ar-init is on but {args.ar_init_ckpt} "
                    f"carries no 'vq'/'vq_ema'; keeping the VQ from --vq-ckpt."
                )

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

    train_dataset = build_imagenet_dataset(
        args.data_dir, image_size=args.image_size, random_hflip=args.random_hflip,
    )
    local_bs = int(args.batch_size // accelerator.num_processes)

    # Deterministic, communication-free shuffling across ranks.
    # We MUST use an explicit DistributedSampler here (not ``shuffle=True``)
    # because Accelerator(rng_types=[]) disables accelerate's per-epoch
    # broadcast of the RNG state. With DistributedSampler each rank
    # computes the *same* permutation from (seed, epoch) deterministically
    # and then takes its own non-overlapping shard -- no collective is
    # required at epoch boundary, which eliminates the 4995-step deadlock
    # (see comment on ``rng_types=[]`` above for the full diagnosis).
    base_seed = args.seed if args.seed is not None else 0
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=base_seed,
        drop_last=True,
    )

    # ``persistent_workers=True`` is also required: without it every epoch
    # boundary triggers a fork/kill of ``num_workers`` worker processes per
    # rank; on 64 ranks that's a global fork storm that itself causes
    # rank-skew large enough to deadlock the next NCCL collective.
    train_loader = DataLoader(
        train_dataset, batch_size=local_bs, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    if accelerator.is_main_process:
        log.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")

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

    # ----- torch.compile -----------------------------------------------------
    # Compile only the AR model (the only trained module here). VQ.encode is
    # called as a method (not __call__), so OptimizedModule proxying gives
    # no speedup; leave VQ untouched.
    if args.compile:
        ar = torch.compile(ar, backend="inductor", mode="default")
        vq = torch.compile(vq, backend="inductor", mode="default")

    # NOTE: ``train_loader`` is deliberately NOT passed to
    # ``accelerator.prepare``. With our explicit ``DistributedSampler`` we
    # already handle sharding + per-epoch shuffle deterministically; letting
    # accelerate wrap the DataLoader as ``DataLoaderShard`` would in some
    # accelerate versions replace our sampler with a ``BatchSamplerShard``
    # (PyTorch shows ``sampler == SequentialSampler`` whenever a custom
    # batch_sampler is set, so we cannot tell from outside which sampler is
    # actually driving iteration). Keeping the raw PyTorch DataLoader
    # guarantees: (a) ``train_sampler.set_epoch(epoch)`` below actually
    # affects the sampler that iterates, (b) no hidden NCCL collective in
    # ``__iter__``, (c) one less moving part to debug. We already move
    # tensors to ``device`` manually inside the loop, so we don't lose
    # ``accelerator``'s auto device placement either.
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
        range(args.max_train_steps), initial=global_step, desc="Step",
        disable=not accelerator.is_local_main_process,
    )

    # Start at the epoch that matches ``global_step``: each rank sees the
    # same number of optimizer steps per epoch (``len(train_loader)`` is
    # identical on all ranks because DistributedSampler+drop_last yields
    # the same per-rank batch count), so the epoch index is deterministic.
    epoch = global_step // max(len(train_loader), 1)
    while True:
        # set_epoch is what makes DistributedSampler reshuffle deterministically
        # across ranks; MUST be called before iterating each new epoch.
        train_sampler.set_epoch(epoch)
        for raw_image, y in train_loader:
            raw_image = raw_image.to(device)
            labels = y.to(device)
            processed = preprocess_imgs_for_codec(raw_image)
            B = processed.shape[0]

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
                        idx=z_indices[:, :-1], cond_idx=labels,
                        return_hidden_at=args.encoder_depth,
                    )
                else:
                    logits = ar(idx=z_indices[:, :-1], cond_idx=labels)
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
                if accelerator.is_main_process:
                    ck = {
                        "ar": _orig(accelerator.unwrap_model(ar)).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": vars(args),
                        "steps": global_step,
                    }
                    if projectors is not None:
                        ck["projectors"] = accelerator.unwrap_model(projectors).state_dict()
                    path = f"{ckpt_dir}/{global_step:07d}.pt"
                    torch.save(ck, path); log.info(f"Saved checkpoint to {path}")

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

                    # (2) AR-generated samples. NOTE: do NOT wrap in
                    # ``accelerator.autocast()`` -- ``setup_caches`` allocates
                    # KVCache buffers in ``tok_embeddings.weight.dtype`` (fp32
                    # under accelerate mixed-precision since params stay fp32),
                    # and bf16 activations would clash with the fp32 in-place
                    # ``k_cache[..., input_pos] = k_val`` write.
                    n_sample = max(1, min(args.sampling_num, labels.shape[0]))
                    c_indices = labels[:n_sample].to(device=device, dtype=torch.long)
                    index_sample = generate(
                        unwrapped_ar, c_indices, latent_size ** 2,
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
                        latent_size, latent_size,
                    ]
                    with torch.no_grad():
                        sample_imgs = unwrapped_vq.decode_code(index_sample, qzshape)
                        sample_imgs = (sample_imgs.clamp(-1, 1) + 1) / 2.0

                    # Drop kv-cache state (next iter's training-mode forward
                    # would otherwise hit the cached K/V and crash).
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
            # Two FID numbers per trigger so the live vs EMA AR comparison is
            # always available side-by-side (mirrors Stage 1's
            # ``fid/{live,ema}`` setup; here VQ is frozen for the whole
            # stage so there is no separate VQ-EMA branch -- both runs
            # share the same ``vq``):
            #
            #   * ``fid/live`` -- live AR + frozen Stage-1 VQ. Tracks the
            #                     actual params being optimized and so
            #                     can be noisy / spike under bad batches.
            #   * ``fid/ema``  -- EMA  AR + the same frozen VQ. EMA lags
            #                     the live AR by ~1/(1-decay) steps and is
            #                     the smoother proxy for inference quality
            #                     at the corresponding step.
            #
            # Sample dirs are step-tagged with ``_live`` / ``_ema`` so the
            # two runs don't clobber each other when ``--eval-keep-samples``
            # is on.
            do_eval = (
                args.eval_steps > 0
                and args.fid_reference_file
                and accelerator.sync_gradients
                and (global_step % args.eval_steps == 0 and global_step > 0)
            )
            if do_eval:
                unwrapped_ar_eval = _orig(accelerator.unwrap_model(ar))
                prev_ar_training = unwrapped_ar_eval.training
                unwrapped_ar_eval.eval()
                # ``vq`` is already frozen in eval (set up at init); no
                # state to touch on its side.

                fid_kwargs = dict(
                    accelerator=accelerator,
                    vq_for_sampling=vq,  # frozen, never DDP/compile-wrapped here
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

                # ------ Run #1: live AR + frozen VQ -> fid_live -----------
                # No parameter swap needed -- the DDP/compile-wrapped
                # ``ar`` already holds live params.
                eval_sample_dir_live = os.path.join(
                    args.output_dir, args.exp_name, "eval_samples",
                    f"{global_step:07d}_live",
                )
                fid_live = run_distributed_fid_eval(
                    ar_for_sampling=unwrapped_ar_eval,
                    sample_dir=eval_sample_dir_live,
                    **fid_kwargs,
                )

                # ------ Run #2: EMA AR + frozen VQ -> fid_ema -------------
                # Stash live AR params (CPU clones to avoid GPU spike) and
                # swap in EMA params on the underlying nn.Module so the
                # same wrapped ``ar(...)`` call now uses EMA weights.
                stashed = [p.detach().cpu().clone() for p in unwrapped_ar_eval.parameters()]
                for ema_p, mp in zip(ema.parameters(), unwrapped_ar_eval.parameters()):
                    mp.data.copy_(ema_p.to(mp.device).data)

                eval_sample_dir_ema = os.path.join(
                    args.output_dir, args.exp_name, "eval_samples",
                    f"{global_step:07d}_ema",
                )
                fid_ema = run_distributed_fid_eval(
                    ar_for_sampling=unwrapped_ar_eval,
                    sample_dir=eval_sample_dir_ema,
                    **fid_kwargs,
                )

                # Restore live AR params and prior training mode.
                for sp, mp in zip(stashed, unwrapped_ar_eval.parameters()):
                    mp.data.copy_(sp.to(mp.device).data)
                if prev_ar_training:
                    unwrapped_ar_eval.train()
                else:
                    unwrapped_ar_eval.eval()

                if accelerator.is_main_process:
                    # Namespace under ``fid/`` so the two curves group
                    # together in the wandb panel sidebar.
                    fid_logs = {}
                    if fid_live is not None:
                        fid_logs["fid/live"] = fid_live
                    if fid_ema is not None:
                        fid_logs["fid/ema"] = fid_ema
                    if fid_logs:
                        accelerator.log(fid_logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break
        epoch += 1

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

    # Online FID eval. See train_gear.py for matching docstring.
    p.add_argument("--eval-steps", type=int, default=0,
                   help="Run an EMA-based FID eval every N global steps. "
                        "Set <= 0 (default) to disable. Independent from "
                        "--sampling-steps which only logs a few wandb images.")
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
        help="When the VQ source (--vq-ckpt, or --ar-init-ckpt if "
             "--vq-override-from-ar-init is on) is one of our Stage-0/1 ckpts "
             "(which carry both `vq` and `vq_ema`), load the EMA shadow "
             "instead of the live params. EMA is what Stage 1 reports under "
             "`vq_val_ema/*` and is the recommended frozen tokenizer for "
             "Stage 2 (default). Pass --no-vq-use-ema for the live `vq`. "
             "Falls back silently to whichever tensor is present. Has no "
             "effect on legacy `{model: ...}` ckpts (e.g. `vq_ds16_c2i.pt`) "
             "or MAGVIT2 lightning ckpts (those expose only one source).",
    )
    p.add_argument(
        "--vq-override-from-ar-init",
        action=argparse.BooleanOptionalAction, default=True,
        help="If --ar-init-ckpt is a Stage-1 ckpt that bundles a co-trained "
             "tokenizer (`vq`/`vq_ema`), re-load the frozen VQ from THAT ckpt "
             "(overriding --vq-ckpt) so the tokenizer matches the AR's "
             "training-time token statistics. Respects --vq-use-ema. On by "
             "default; pass --no-vq-override-from-ar-init to keep the "
             "--vq-ckpt tokenizer regardless.",
    )

    # AR
    p.add_argument("--ar-model", type=str, default="LlamaGen-B",
                   choices=list(LlamaGen_models.keys()))
    p.add_argument("--ar-init-ckpt", type=str, default=None)
    p.add_argument("--cls-token-num", type=int, default=1)
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
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--max-train-steps", type=int, default=1500000)
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
