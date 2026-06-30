"""REPA-style encoders + small image helpers.

Trimmed-down version of REPA-E `utils.py` keeping only what's needed for the
VQ-AR Stage 1 / Stage 2 trainers. The REPA target-encoder family selection
(DINOv2 / DINOv3 / V-JEPA 2.1 / CLIP) is delegated to
:mod:`src.repa_encoders`; this module just re-exports the three
top-level entry points (``load_encoders`` / ``preprocess_raw_image`` /
``extract_repa_target``) for backwards compatibility with the trainers.

Also hosts ``run_distributed_fid_eval`` -- the shared online FID evaluator
called from both ``train_gear.py`` and ``train_ar.py`` to compute FID
every ``--eval-steps`` steps using the EMA AR + a (frozen / current) VQ.
"""

import concurrent.futures
import io
import math
import os
import shutil
from functools import partial
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from src.repa_encoders import (
    extract_at_layer,
    extract_repa_target,
    load_encoders,
    num_encoder_layers,
    preprocess_raw_image,
    spatial_norm,
    supports_resolution,
)

__all__ = [
    # Re-exported from repa_encoders so downstream `from src.utils import
    # load_encoders` keeps working unchanged.
    "extract_at_layer",
    "extract_repa_target",
    "load_encoders",
    "num_encoder_layers",
    "preprocess_raw_image",
    "spatial_norm",
    "supports_resolution",
    # Original utils in this module.
    "get_constant_schedule_with_warmup",
    "preprocess_imgs_for_codec",
    "count_trainable_params",
    "center_crop_arr",
    "run_distributed_fid_eval",
    "run_vq_reconstruction_eval",
    "load_pretrained_tokenizer_state_dict",
]


# =============================================================================
# Learning-rate schedules
# =============================================================================
def _constant_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int,
) -> float:
    """Linear warmup from 0 to 1 over ``num_warmup_steps``, then constant 1.

    Edge cases:
        * ``num_warmup_steps == 0`` -> returns 1.0 from step 0 (no warmup).
        * ``num_warmup_steps == 1`` -> step 0 returns 0.0, step 1+ returns 1.0.
          Combined with ``optimizer.step()`` semantics (lr applied BEFORE the
          step), the first parameter update is a no-op (AdamW: ``-lr*g/...``
          = 0 when lr=0). Useful trick: an eval at ``global_step==1`` then
          shows the *unmodified* pretrained baseline.
    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps))
    return 1.0


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """Linear warmup -> constant lr (no decay).

    Mirrors ``transformers.get_constant_schedule_with_warmup``. Returns a
    ``LambdaLR`` so its state (``last_epoch``) can be saved/restored via the
    standard ``state_dict()`` / ``load_state_dict()`` protocol.
    """
    return LambdaLR(
        optimizer,
        partial(
            _constant_schedule_with_warmup_lr_lambda,
            num_warmup_steps=int(num_warmup_steps),
        ),
        last_epoch=last_epoch,
    )


def preprocess_imgs_for_codec(imgs):
    """uint8 [0, 255] -> float32 [-1, 1]. Same as REPA-E `preprocess_imgs_vae`."""
    return imgs.float() / 127.5 - 1.0


def count_trainable_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def center_crop_arr(image_arr, image_size):
    """ADM-style center crop.

    Accepts either a ``PIL.Image.Image`` or a ``(H, W, 3) uint8`` numpy array
    and always returns a ``(image_size, image_size, 3) uint8`` numpy array,
    matching the REPA-E preprocessing contract.

    Ref: https://github.com/openai/guided-diffusion/blob/8fb3ad9/guided_diffusion/image_datasets.py#L126
    """
    pil_image = image_arr if isinstance(image_arr, Image.Image) else Image.fromarray(image_arr)

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


# =============================================================================
# Online FID evaluation
# =============================================================================
def _save_png_bytes(path: str, png_bytes: bytes) -> None:
    """Write pre-encoded PNG bytes to disk.

    The PIL encode happens upstream (in the worker thread, not the main
    thread that just kicked off the next CUDA generate) so this function
    is *purely* a write-syscall wrapper. Splitting encode/write into two
    functions makes it easy to swap in a different storage backend
    (S3, ramdisk, etc.) later without touching the GPU loop.
    """
    with open(path, "wb") as f:
        f.write(png_bytes)


def _encode_and_save(path: str, sample_hwc_uint8: np.ndarray) -> None:
    """PIL-encode a ``(H, W, 3) uint8`` array and write the PNG to disk.

    Runs inside a ``ThreadPoolExecutor`` worker -- both ``Image.save``
    (zlib compression) and ``open(...).write(...)`` release the GIL, so
    the main thread is free to launch the next batch's CUDA work while
    these encode / write in parallel.
    """
    buf = io.BytesIO()
    Image.fromarray(sample_hwc_uint8).save(buf, format="PNG")
    _save_png_bytes(path, buf.getvalue())


@torch.no_grad()
def run_distributed_fid_eval(
    *,
    accelerator,
    ar_for_sampling,
    vq_for_sampling,
    sample_dir,
    fid_num,
    fid_reference_file,
    num_classes,
    latent_size,
    codebook_embed_dim,
    eval_per_proc_batch_size,
    sampling_temperature=1.0,
    sampling_top_k=0,
    sampling_top_p=1.0,
    sampling_cfg_scale=1.0,
    sampling_cfg_interval=-1,
    fid_batch_size=200,
    fid_dims=2048,
    fid_num_workers=8,
    save_workers=8,
    keep_samples=False,
    log=None,
):
    """Distributed sampling + (rank-0) FID computation.

    Mirrors ``UniWorld-V1-Backup/inference.py`` 's distributed sampling layout
    (each rank generates ``eval_per_proc_batch_size`` images at a time,
    rank-stamped filenames so PNGs cover ``[0, total_samples)`` contiguously)
    and ``UniWorld-V1-Backup/train.py:346-362`` 's main-rank-only FID call,
    fused into a single helper that both Stage-1 and Stage-2 trainers use.

    Parameters
    ----------
    accelerator
        The current ``accelerate.Accelerator``. Used for ``device``,
        ``process_index``, ``num_processes``, ``wait_for_everyone()``.
    ar_for_sampling
        The (already-unwrapped, uncompiled, eval-mode) AR ``nn.Module``.
        EMA params are expected to already be loaded into it. We never wrap
        ``generate()`` in an autocast context: the KV-cache buffers
        ``setup_caches`` allocates inherit ``tok_embeddings.weight.dtype``
        (= fp32 under accelerate mixed-precision since params stay fp32),
        and bf16/fp16 attention activations would clash with the fp32
        in-place ``k_cache[..., input_pos] = k_val`` write.
    vq_for_sampling
        The (already-unwrapped, uncompiled, eval-mode) VQ ``nn.Module``.
        Provides ``decode_code(index_sample, qzshape)`` returning images
        in ``[-1, 1]``.
    sample_dir
        Where this eval call writes its PNGs (rank 0 mkdirs). Should be a
        per-step directory so concurrent / repeated evals don't collide.
        Deleted at the end unless ``keep_samples=True``.
    fid_num
        Target number of samples for FID, e.g. 50_000 (canonical) or
        10_000 (cheap online proxy).
    fid_reference_file
        Path to a precomputed ``.npz`` containing ``mu`` / ``sigma`` over
        the reference distribution (e.g. ``VIRTUAL_imagenet256_labeled.npz``).
    num_classes, latent_size, codebook_embed_dim
        Sampling shape parameters; class indices are uniform random in
        ``[0, num_classes)``.
    eval_per_proc_batch_size
        Per-rank batch size used during the sampling loop. Bigger = faster
        but more peak GPU memory.
    sampling_*
        AR sampling controls. Defaults are "no CFG" + greedy-ish
        (top-k=0, top-p=1.0, temperature=1.0) -- i.e. pure ``argmax`` of the
        softened logits, matching the user's request to FID-eval *without*
        CFG.
    fid_batch_size, fid_dims, fid_num_workers
        Forwarded to ``calculate_fid_given_paths``.
    save_workers
        Per-rank thread-pool size used to overlap PIL PNG encode + disk
        write with the next batch's CUDA generate. Both ``Image.save``
        (zlib) and the kernel write release the GIL, so threads scale
        well even though Python has a global lock. Set to 0 to disable
        threading and save synchronously (debug).
    keep_samples
        If False (default), the PNG folder is removed once FID is done. Set
        True to keep snapshots for offline inspection.
    log
        Optional ``logging.Logger`` for status messages (rank 0 only).

    Returns
    -------
    float | None
        The FID value on rank 0; ``None`` on other ranks.
    """
    # Local imports keep the module importable even when these heavy
    # dependencies aren't on the path (e.g. when only ``load_encoders`` is
    # used at preprocessing time).
    from models.generate import generate  # type: ignore  # noqa: WPS433

    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    # Round ``fid_num`` up so every rank does the same number of iters.
    n = int(eval_per_proc_batch_size)
    global_batch = n * world_size
    total_samples = int(math.ceil(fid_num / global_batch) * global_batch)
    iters = total_samples // global_batch

    if accelerator.is_main_process:
        os.makedirs(sample_dir, exist_ok=True)
        if log is not None:
            log.info(
                f"[FID] target={fid_num} sampling={total_samples} "
                f"(iters={iters} x global_batch={global_batch}) -> {sample_dir}"
            )
    accelerator.wait_for_everyone()

    # Sampling loop (no autocast, see docstring). We deliberately do NOT
    # use a tqdm bar to keep multi-node logs quiet; a 64-rank cluster
    # generating 50k samples is < ~1 min anyway.
    #
    # Threading rationale:
    #   * AR sampling + VQ decode are the *real* GPU work. They're already
    #     low-SM-occupancy because `generate()` is autoregressive (256
    #     small-batch decode steps), so we want zero CPU stalls between
    #     consecutive GPU launches.
    #   * The per-batch tensor->numpy `.to("cpu")` is a hard CUDA sync.
    #     Synchronous PIL saves *after* the sync used to add ~10-15% wall
    #     time on top of `generate()`, during which the GPU sits idle.
    #   * Solution: dispatch encode+write to a per-rank ThreadPoolExecutor
    #     and immediately re-enter the loop. PIL releases the GIL during
    #     zlib compression and the kernel write call, so the next iter's
    #     CUDA work runs in parallel with the previous iter's saves.
    executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=int(save_workers))
        if save_workers and save_workers > 0
        else None
    )
    pending = []
    total = 0
    for _ in range(iters):
        c_indices = torch.randint(0, int(num_classes), (n,), device=device)
        qzshape = [n, int(codebook_embed_dim), int(latent_size), int(latent_size)]
        index_sample = generate(
            ar_for_sampling, c_indices, latent_size ** 2,
            cfg_scale=sampling_cfg_scale,
            cfg_interval=sampling_cfg_interval,
            temperature=sampling_temperature,
            top_k=sampling_top_k,
            top_p=sampling_top_p,
            sample_logits=True,
        )
        samples = vq_for_sampling.decode_code(index_sample, qzshape)
        samples = (
            torch.clamp(127.5 * samples + 128.0, 0, 255)
            .permute(0, 2, 3, 1)
            .to("cpu", dtype=torch.uint8)
            .numpy()
        )
        for i in range(samples.shape[0]):
            index = i * world_size + rank + total
            if index >= fid_num:
                continue
            path = f"{sample_dir}/{index:06d}.png"
            # Slice once; numpy.ascontiguousarray makes the buffer
            # standalone so the worker thread doesn't accidentally share
            # storage with the next iter's `samples` array.
            arr = np.ascontiguousarray(samples[i])
            if executor is not None:
                pending.append(executor.submit(_encode_and_save, path, arr))
            else:
                _encode_and_save(path, arr)
        total += global_batch

    # Drain background saves before the rank-0 InceptionV3 pass walks the
    # folder. ``concurrent.futures.wait`` returns as soon as everything is
    # done; we then surface any worker exceptions by re-raising the first
    # one (otherwise FID would silently see a partial sample folder).
    if executor is not None:
        concurrent.futures.wait(pending)
        for fut in pending:
            exc = fut.exception()
            if exc is not None:
                executor.shutdown(wait=False)
                raise exc
        executor.shutdown(wait=True)

    accelerator.wait_for_everyone()

    fid_value = None
    if accelerator.is_main_process:
        # Defer the heavy import (loads InceptionV3 weights) to the rank
        # that actually needs it. ``compute_statistics_of_path`` reads PNGs
        # straight from the folder and the reference path is a precomputed
        # ``mu`` / ``sigma`` ``.npz``, so we don't bother packing an .npz
        # of generated samples here.
        from tools.calculate_fid import calculate_fid_given_paths  # type: ignore  # noqa: WPS433

        if log is not None:
            log.info(f"[FID] Computing FID with {fid_num} samples vs {fid_reference_file}")
        fid_value = float(calculate_fid_given_paths(
            [fid_reference_file, sample_dir],
            batch_size=int(fid_batch_size),
            dims=int(fid_dims),
            device=device,
            num_workers=int(fid_num_workers),
            sp_len=int(fid_num),
        ))
        if log is not None:
            log.info(f"[FID] FID = {fid_value:.4f}")

        if not keep_samples:
            try:
                shutil.rmtree(sample_dir)
            except OSError as exc:
                if log is not None:
                    log.info(f"[FID] failed to clean {sample_dir}: {exc}")

    accelerator.wait_for_everyone()
    return fid_value


# =============================================================================
# Online VQ reconstruction metrics (L1 / PSNR / SSIM / FID over ImageNet val)
# =============================================================================
@torch.no_grad()
def run_vq_reconstruction_eval(
    *,
    accelerator,
    vq,
    val_loader,
    inception=None,
    eval_l1=True,
    eval_psnr=True,
    eval_ssim=True,
    log=None,
):
    """Distributed VQ reconstruction eval over an ImageNet val loader.

    Inspired by ``T2/train_ae_v3.py:valid``. Stage 1 trains the VQ
    end-to-end so we want PSNR / SSIM / FID curves vs a *fixed* held-out
    set to track the actual reconstruction quality rather than the
    train-batch loss (which is noisy and biased by the current sample
    distribution).

    Each rank processes its ``DistributedSampler`` shard and accumulates
    per-batch L1 / PSNR (already global-mean compatible -- linear in
    samples) and per-image SSIM (skimage, CPU). FID requires the full
    feature distribution: each rank cats its local InceptionV3 features
    and ``accelerator.gather`` collects the global tensor before rank 0
    computes ``mu`` / ``sigma`` and the Frechet distance.

    The val loader's ``DistributedSampler`` MUST be built with
    ``shuffle=False, drop_last=False`` -- this gives every rank exactly
    ``ceil(N / world_size)`` samples (with a tiny number of duplicate
    samples padded onto the last few ranks), so ``accelerator.gather``
    works without size-mismatch errors.

    Parameters
    ----------
    vq
        The (unwrapped, uncompiled) VQ ``nn.Module``. Caller is
        responsible for setting ``eval()`` mode and stashing/restoring
        ``train()`` afterwards.
    val_loader
        Plain ``torch.utils.data.DataLoader`` over the val dataset. Must
        yield ``(uint8 (B, 3, H, W), label)`` tuples (matches the train
        loader's contract via :func:`build_imagenet_val_dataset`).
    inception
        Optional ``InceptionV3`` ``nn.Module`` already on ``device`` and
        in ``eval()``. Pass ``None`` to skip FID.
    eval_l1, eval_psnr, eval_ssim
        Toggle individual metrics. SSIM via skimage is the slowest
        (~50-100ms/image, CPU-bound); turn it off if eval cadence is
        tight. Defaults all True.
    log
        Optional logger; rank-0-only summary line at the end.

    Returns
    -------
    dict[str, float]
        Wandb-ready scalar dict, keyed under ``vq_val/<metric>``.
        Metrics not requested / not enabled are simply absent from the
        dict; the caller can pass it straight to ``accelerator.log``.
    """
    device = accelerator.device

    l1_per_batch = []
    psnr_per_batch = []
    ssim_local = []
    feats_ref_local = []
    feats_rec_local = []

    if log is not None:
        log.info(
            f"[VQ-eval] running on val loader (per-rank batches="
            f"{len(val_loader)}, FID={'on' if inception is not None else 'off'})"
        )

    for raw_image, _label in val_loader:
        raw_image = raw_image.to(device, non_blocking=True)
        # uint8 [0, 255] -> float [-1, 1]; mirrors the train-time path so
        # the VQ sees the exact same input distribution at eval time.
        processed = preprocess_imgs_for_codec(raw_image)

        with accelerator.autocast():
            out = vq(processed)
        # vq forward signature is (recon, z_q, vq_losses, info[, d]).
        # Index 0 is always the reconstruction.
        recon = out[0] if isinstance(out, (tuple, list)) else out

        recon = recon.clamp(-1, 1).float()
        recon_01 = (recon + 1.0) / 2.0
        labels_01 = (processed.clamp(-1, 1).float() + 1.0) / 2.0

        if eval_l1:
            l1_per_batch.append((recon_01 - labels_01).abs().mean())

        if eval_psnr:
            mse = ((labels_01 - recon_01) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
            psnr_per_batch.append((20.0 * torch.log10(1.0 / torch.sqrt(mse))).mean())

        if eval_ssim:
            # skimage SSIM is CPU + numpy. Safe to run per-image: with
            # 50k val images / 64 ranks => ~782 images/rank, ~30-60s.
            from skimage.metrics import structural_similarity as ssim_fn  # noqa: WPS433

            labels_np = labels_01.detach().cpu().numpy()
            recon_np = recon_01.detach().cpu().numpy()
            for i in range(labels_np.shape[0]):
                ssim_local.append(
                    float(ssim_fn(
                        labels_np[i], recon_np[i],
                        channel_axis=0, data_range=1.0,
                    ))
                )

        if inception is not None:
            # Inception expects (B, 3, H, W) in [0, 1]; resizes internally
            # to 299x299 and (re)normalizes to [-1, 1]. We deliberately
            # feed fp32 tensors (BN running stats are fp32 -- mixing in a
            # bf16 input under autocast would dtype-mismatch).
            f_ref = inception(labels_01)[0].squeeze(-1).squeeze(-1)
            f_rec = inception(recon_01)[0].squeeze(-1).squeeze(-1)
            feats_ref_local.append(f_ref.float())
            feats_rec_local.append(f_rec.float())

    metrics: dict = {}

    # Reduce per-rank scalars (DistributedSampler's pad guarantees every
    # rank saw the same number of batches, so a flat mean is correct).
    if eval_l1:
        l1_t = (torch.stack(l1_per_batch).mean()
                if l1_per_batch else torch.zeros((), device=device))
        metrics["vq_val/l1"] = accelerator.gather(l1_t).mean().item()

    if eval_psnr:
        psnr_t = (torch.stack(psnr_per_batch).mean()
                  if psnr_per_batch else torch.zeros((), device=device))
        metrics["vq_val/psnr"] = accelerator.gather(psnr_t).mean().item()

    if eval_ssim:
        ssim_t = torch.tensor(
            float(np.mean(ssim_local)) if ssim_local else 0.0,
            device=device, dtype=torch.float32,
        )
        metrics["vq_val/ssim"] = accelerator.gather(ssim_t).mean().item()

    if inception is not None:
        feats_ref_t = torch.cat(feats_ref_local, dim=0)
        feats_rec_t = torch.cat(feats_rec_local, dim=0)
        gathered_ref = accelerator.gather(feats_ref_t)
        gathered_rec = accelerator.gather(feats_rec_t)
        # All ranks have the global tensor after gather, but only rank 0
        # needs to do the (small) numpy cov + Frechet computation.
        if accelerator.is_main_process:
            from tools.calculate_fid import calculate_frechet_distance  # noqa: WPS433

            ref_np = gathered_ref.cpu().numpy()
            rec_np = gathered_rec.cpu().numpy()
            mu1, mu2 = ref_np.mean(0), rec_np.mean(0)
            sigma1 = np.cov(ref_np, rowvar=False)
            sigma2 = np.cov(rec_np, rowvar=False)
            metrics["vq_val/fid"] = float(
                calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
            )

    if log is not None and accelerator.is_main_process:
        bits = []
        if "vq_val/l1" in metrics:
            bits.append(f"L1={metrics['vq_val/l1']:.4f}")
        if "vq_val/psnr" in metrics:
            bits.append(f"PSNR={metrics['vq_val/psnr']:.2f}")
        if "vq_val/ssim" in metrics:
            bits.append(f"SSIM={metrics['vq_val/ssim']:.4f}")
        if "vq_val/fid" in metrics:
            bits.append(f"FID={metrics['vq_val/fid']:.4f}")
        log.info("[VQ-eval] " + " ".join(bits))

    return metrics


# =============================================================================
# Tokenizer-pretrained-checkpoint loader (used by Stage 0 / 1 / 2)
# =============================================================================
def load_pretrained_tokenizer_state_dict(
    ckpt_path: str,
    map_location: str = "cpu",
    use_ema: bool = False,
) -> "dict":
    """Best-effort extraction of a tokenizer state_dict.

    Stage 0/1 (``--vq-pretrained-ckpt``) and Stage 2 (``--vq-ckpt``) all
    funnel through this helper so the same path string works regardless
    of the checkpoint's origin. Supported wrappings:

      * ``{"vq": state_dict, "vq_ema": state_dict, ...}`` -- our Stage-0 /
        Stage-1 ckpts. ``vq`` is the live optimised state at the
        ``global_step`` of the ckpt; ``vq_ema`` is the EMA shadow that
        Stage 1 reports under ``vq_val_ema/*``. Pass ``use_ema=True`` to
        load the EMA copy (recommended for downstream Stage-2 / inference
        use, since it lags the live params by ~1/(1-decay) steps and is
        empirically smoother). Falls back to ``vq`` if the ckpt is from
        an older trainer that didn't save ``vq_ema``.
      * ``{"model": state_dict}``                      -- legacy LlamaGen ``vq_ds16_c2i.pt``.
      * ``{"state_dict": {... model_ema.*  ...}}``     -- TencentARC Open-MAGVIT2
        Lightning ckpt (LFQ). Routed through
        :func:`models.lfq_model.convert_magvit2_pretrained_state_dict`
        which prefers the EMA shadow (matches the MAGVIT2 reference
        inference recipe). ``use_ema`` is ignored here: MAGVIT2's
        EMA-vs-live selection is governed by the converter's own
        ``prefer_ema`` flag (always True), since the LitEma shadow is
        the only sensible source for inference / fine-tuning warm starts.
      * Bare state_dict (no wrapping)                  -- accepted as-is.

    Returns a flat ``str -> Tensor`` mapping suitable for
    ``model.load_state_dict(..., strict=False)``.
    """
    raw = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected dict-like checkpoint at {ckpt_path}, got {type(raw)}."
        )

    # MAGVIT2 lightning: top-level "state_dict" containing both live and EMA
    # params. The LitEma shadow is identifiable by the "model_ema." prefix
    # with dots stripped from the inner key path.
    if "state_dict" in raw and isinstance(raw["state_dict"], dict):
        inner = raw["state_dict"]
        if any(k.startswith("model_ema.") for k in inner):
            # IBQ ships a learnable codebook (``quantize.embedding.*``) plus
            # ``quant_conv`` / ``post_quant_conv``; the single-codebook
            # MAGVIT2 LFQ ckpt has none of these. Dispatch on that signature
            # so the same --vq-pretrained-ckpt flag works for both families.
            if any("quantize.embedding" in k for k in inner):
                from models.ibq_model import (  # noqa: E402
                    convert_ibq_pretrained_state_dict,
                )
                return convert_ibq_pretrained_state_dict(inner, prefer_ema=True)
            # Local import to avoid pulling models/* at module load time.
            from models.lfq_model import (  # noqa: E402
                convert_magvit2_pretrained_state_dict,
            )
            return convert_magvit2_pretrained_state_dict(inner, prefer_ema=True)
        return inner

    # Our own ckpts (Stage 0 / Stage 1): may carry both ``vq`` and ``vq_ema``.
    if "vq" in raw or "vq_ema" in raw:
        if use_ema and "vq_ema" in raw and isinstance(raw["vq_ema"], dict):
            return raw["vq_ema"]
        # use_ema requested but ckpt has no vq_ema -> silent fall-through
        # to the live ``vq``. (No logger here -- this helper is called
        # from many places; the trainer-side caller logs the source.)
        if "vq" in raw and isinstance(raw["vq"], dict):
            return raw["vq"]
    if "model" in raw and isinstance(raw["model"], dict):
        return raw["model"]
    return raw
