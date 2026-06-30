"""Open-MAGVIT2 LFQ tokenizer ported to plain PyTorch.

The reference implementation lives at
``SEED-Voken/src/Open_MAGVIT2/models/lfqgan_pretrain.py`` and
``SEED-Voken/src/Open_MAGVIT2/modules/{diffusionmodules,vqvae}/`` and is
written for PyTorch Lightning + LitEma. This module strips Lightning out
and exposes an ``nn.Module`` whose API mirrors
``models/vq_model.VQModel`` so it drops into the same Stage 0/1/2
training scripts without code changes.

API contract (matches ``models.vq_model.VQModel`` so train_tokenizer/gear/ar can
swap one for the other through a ``--vq-model`` flag):

* ``LFQ_16(codebook_size, codebook_embed_dim, ...) -> LFQModel``
* ``model.encode(x)``                 -> ``(quant, diff, info)``
* ``model.encode(x, return_distance=True)``
                                      -> ``(quant, diff, info, d)``
* ``model.decode(quant)``             -> recon
* ``model.decode_code(idx, shape)``   -> recon (shape = (B, D, H, W))
* ``model(input)``                    -> ``(recon, diff)``
* ``model(input, return_distance=True)``
                                      -> ``(recon, quant, diff, info, d)``

where ``diff = (vq_loss, commit_loss, entropy_loss_tuple,
codebook_usage_zero)`` and ``info = (perp_or_None,
min_encodings_or_None, min_encoding_indices)`` -- the slot shapes /
semantics match ``VectorQuantizer`` so the outer
``ReconstructionLossVQ`` / train loops are byte-compatible.

Pretrained-weight loading: ``convert_magvit2_pretrained_state_dict``
takes a TencentARC Open-MAGVIT2 Lightning ``state_dict`` (which carries
both live ``encoder.*`` / ``decoder.*`` params and ``model_ema.<key
with dots stripped>`` LitEma shadows) and returns a state_dict that
``LFQModel.load_state_dict(..., strict=False)`` accepts.

Single-codebook LFQ has no learnable codebook (the codebook is the
``{-1,+1}^D`` hypercube, registered as a non-persistent buffer); the
only trainable params live in the encoder/decoder. ``embed_dim`` is
therefore forced to ``log2(codebook_size)``.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from math import log2
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


# =============================================================================
# Encoder / Decoder (ported verbatim from
# SEED-Voken/src/Open_MAGVIT2/modules/diffusionmodules/improved_model.py).
# Kept structurally identical so the TencentARC pretrained state_dict drops in.
# =============================================================================
def swish(x):
    return x * torch.sigmoid(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        use_conv_shortcut: bool = False,
        use_agn: bool = False,
    ) -> None:
        super().__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_conv_shortcut = use_conv_shortcut
        self.use_agn = use_agn

        if not use_agn:
            self.norm1 = nn.GroupNorm(32, in_filters, eps=1e-6)
        self.norm2 = nn.GroupNorm(32, out_filters, eps=1e-6)

        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, bias=False)

        if in_filters != out_filters:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1, bias=False)
            else:
                self.nin_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        residual = x
        if not self.use_agn:
            x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_filters != self.out_filters:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)
        return x + residual


def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Depth-to-Space DCR mode (depth-column-row). Lifted verbatim from the
    MAGVIT2 reference -- we only need the 2-D case here.
    """
    if x.dim() < 3:
        raise ValueError("Expecting a channels-first (*CHW) tensor of at least 3 dimensions")
    c, h, w = x.shape[-3:]
    s = block_size ** 2
    if c % s != 0:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c}"
        )
    outer_dims = x.shape[:-3]
    x = x.view(-1, block_size, block_size, c // s, h, w)
    x = x.permute(0, 3, 4, 1, 5, 2)
    x = x.contiguous().view(*outer_dims, c // s, h * block_size, w * block_size)
    return x


class Upsampler(nn.Module):
    """Pixel-shuffle-style upsampler used by the MAGVIT2 decoder."""

    def __init__(self, dim: int, dim_out: Optional[int] = None):
        super().__init__()
        del dim_out
        dim_out = dim * 4
        self.conv1 = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)
        self.depth2space = depth_to_space

    def forward(self, x):
        out = self.conv1(x)
        out = self.depth2space(out, block_size=2)
        return out


class AdaptiveGroupNorm(nn.Module):
    """Conditioning the decoder GroupNorm on the (pre-quant) latent.

    See improved_model.AdaptiveGroupNorm. ``quantizer`` here is the
    z-channels-shaped pre-quant tensor; its per-channel mean/std
    drive the affine.
    """

    def __init__(self, z_channel: int, in_filters: int, num_groups: int = 32, eps: float = 1e-6):
        super().__init__()
        del num_groups
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_filters, eps=eps, affine=False)
        self.gamma = nn.Linear(z_channel, in_filters)
        self.beta = nn.Linear(z_channel, in_filters)
        self.eps = eps

    def forward(self, x, quantizer):
        b, c, _, _ = x.shape
        scale = rearrange(quantizer, "b c h w -> b c (h w)")
        scale = scale.var(dim=-1) + self.eps  # not unbiased -- matches reference
        scale = scale.sqrt()
        scale = self.gamma(scale).view(b, c, 1, 1)

        bias = rearrange(quantizer, "b c h w -> b c (h w)")
        bias = bias.mean(dim=-1)
        bias = self.beta(bias).view(b, c, 1, 1)

        x = self.gn(x)
        x = scale * x + bias
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        in_channels: int,
        num_res_blocks: int,
        z_channels: int,
        ch_mult: Tuple[int, ...] = (1, 2, 2, 4),
        resolution: int,
        double_z: bool = False,
    ):
        super().__init__()
        del out_ch, resolution, double_z
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(ch_mult)

        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1, bias=False)

        self.down = nn.ModuleList()
        in_ch_mult = (1,) + tuple(ch_mult)
        for i_level in range(self.num_blocks):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResBlock(block_in, block_out))
                block_in = block_out

            down = nn.Module()
            down.block = block
            if i_level < self.num_blocks - 1:
                down.downsample = nn.Conv2d(block_out, block_out, kernel_size=3, stride=2, padding=1)
            self.down.append(down)

        self.mid_block = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in))

        self.norm_out = nn.GroupNorm(32, block_out, eps=1e-6)
        self.conv_out = nn.Conv2d(block_out, z_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        for i_level in range(self.num_blocks):
            for i_block in range(self.num_res_blocks):
                x = self.down[i_level].block[i_block](x)
            if i_level < self.num_blocks - 1:
                x = self.down[i_level].downsample(x)
        for res in range(self.num_res_blocks):
            x = self.mid_block[res](x)
        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        in_channels: int,
        num_res_blocks: int,
        z_channels: int,
        ch_mult: Tuple[int, ...] = (1, 2, 2, 4),
        resolution: int,
        double_z: bool = False,
    ):
        super().__init__()
        del in_channels, resolution, double_z
        self.ch = ch
        self.num_blocks = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[self.num_blocks - 1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, padding=1, bias=True)

        self.mid_block = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in))

        self.up = nn.ModuleList()
        self.adaptive = nn.ModuleList()

        for i_level in reversed(range(self.num_blocks)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            self.adaptive.insert(0, AdaptiveGroupNorm(z_channels, block_in))
            for _ in range(num_res_blocks):
                block.append(ResBlock(block_in, block_out))
                block_in = block_out

            up = nn.Module()
            up.block = block
            if i_level > 0:
                up.upsample = Upsampler(block_in)
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, padding=1)

    @property
    def last_layer(self):
        # Match VQ_model.Decoder API used by `ReconstructionLossVQ`'s
        # adaptive disc-weight (calls `self.last_layer` to get the final
        # generator conv weight for grad-norm comparison).
        return self.conv_out.weight

    def forward(self, z):
        style = z.clone()  # pre-quant latent drives the AdaptiveGroupNorm
        z = self.conv_in(z)
        for res in range(self.num_res_blocks):
            z = self.mid_block[res](z)
        for i_level in reversed(range(self.num_blocks)):
            z = self.adaptive[i_level](z, style)
            for i_block in range(self.num_res_blocks):
                z = self.up[i_level].block[i_block](z)
            if i_level > 0:
                z = self.up[i_level].upsample(z)
        z = self.norm_out(z)
        z = swish(z)
        z = self.conv_out(z)
        return z


# =============================================================================
# Lookup-Free Quantizer (adapted from SEED-Voken's lookup_free_quantize.LFQ).
#
# Differences vs. the reference:
#   * Return tuple shaped to match models/vq_model.py:VectorQuantizer so the
#     outer ReconstructionLossVQ and train loops don't need to special-case.
#   * ``return_distance=True`` exposes a (B*H*W, K) "distance" tensor whose
#     ``softmax(-d / T)`` gives the standard LFQ soft assignment over the
#     full 2^D codebook -- this is exactly what src/train_gear.py
#     consumes for soft-label AR training.
#   * Single-codebook only (``num_codebooks=1``). Multi-codebook / token
#     factorization is out of scope and would change the indices layout
#     contract that AR consumes.
# =============================================================================
class LFQQuantizer(nn.Module):
    """Lookup-free quantizer with the API of ``VectorQuantizer``.

    The codebook is the static ``{-1, +1}^codebook_dim`` hypercube; there
    are no learnable parameters. The forward computes:

      * ``quantized`` = sign(z) with a straight-through gradient.
      * ``indices``   = bit-pack of ``(quantized > 0)`` along the channel
        dim (big-endian) -> integer in [0, codebook_size).
      * Entropy auxiliary loss + commit MSE during training (zeros at eval).
    """

    def __init__(
        self,
        codebook_size: int,
        embed_dim: Optional[int] = None,
        sample_minimization_weight: float = 1.0,
        batch_maximization_weight: float = 1.0,
        entropy_loss_ratio: float = 1.0,
        entropy_loss_temperature: float = 0.01,
        commit_loss_beta: float = 0.25,
    ):
        super().__init__()
        cb_dim = int(log2(codebook_size))
        assert 2 ** cb_dim == codebook_size, (
            f"codebook_size must be a power of 2 for LFQ (got {codebook_size})."
        )
        if embed_dim is None:
            embed_dim = cb_dim
        assert embed_dim == cb_dim, (
            f"For single-codebook LFQ, embed_dim must equal log2(codebook_size)."
            f" Got embed_dim={embed_dim}, log2(codebook_size)={cb_dim}."
        )

        self.codebook_size = codebook_size
        self.codebook_dim = cb_dim
        self.embed_dim = embed_dim
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight
        # Mutable from outside (Stage 0/1 set it from CLI args, mirroring how
        # VectorQuantizer.entropy_loss_ratio is overridden).
        self.entropy_loss_ratio = float(entropy_loss_ratio)
        self.entropy_loss_temperature = float(entropy_loss_temperature)
        # Pre-multiplier on the commit MSE returned to the outer loss helper.
        # Mirrors ``VectorQuantizer.beta`` (0.25 default) so that
        # ``losses.py`` -- which assumes ``commit_loss`` is already
        # quantizer-scaled and only multiplies by ``quantizer_weight=1.0`` --
        # produces the SAME effective ``0.25 * MSE(z_e, sg(z_q))`` term used
        # by both the LlamaGen VQ-VAE recipe and the MAGVIT2 LFQ pretrain
        # recipe (their loss config has ``commit_weight: 0.25``). A naive
        # port that returned the raw MSE would silently multiply commit by
        # 4x, which on a warm-started LFQ checkpoint manifests as the
        # initial ``commit_loss`` spiking to ~0.4 then collapsing to ~0.05
        # (the encoder being yanked to a sharper {-1, +1} regime than it
        # was pretrained for).
        self.commit_loss_beta = float(commit_loss_beta)

        # Bit mask for index <-> bit conversions.
        self.register_buffer(
            "mask", 2 ** torch.arange(cb_dim), persistent=False,
        )
        # Pre-materialized {-1, +1}^D codebook for the entropy loss logits.
        all_codes = torch.arange(codebook_size)
        bits = self._indices_to_bits(all_codes, cb_dim)
        codebook = bits.float() * 2.0 - 1.0  # (K, D)
        self.register_buffer("codebook", codebook, persistent=False)

    # ---- bit / index conversions ------------------------------------------
    @staticmethod
    def _indices_to_bits(x: torch.Tensor, codebook_dim: int) -> torch.Tensor:
        mask = 2 ** torch.arange(codebook_dim, device=x.device, dtype=torch.long)
        return (x.unsqueeze(-1) & mask) != 0

    def indices_to_bits(self, x: torch.Tensor) -> torch.Tensor:
        return self._indices_to_bits(x, self.codebook_dim)

    # ---- decode (indices -> dense {-1, +1}^D vectors) ---------------------
    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert long-tensor of indices (any shape) to {-1, +1} vectors
        with one extra trailing dim of size ``codebook_dim``.
        """
        bits = self.indices_to_bits(indices)
        return bits.to(self.codebook.dtype) * 2.0 - 1.0

    def get_codebook_entry(
        self,
        indices: torch.Tensor,
        shape: Optional[Tuple[int, int, int, int]] = None,
        channel_first: bool = True,
    ) -> torch.Tensor:
        """Mirror ``VectorQuantizer.get_codebook_entry`` so ``decode_code``
        in the parent ``LFQModel`` reads the same way as in ``VQModel``.

        Parameters
        ----------
        indices : LongTensor
            Either flat ``(B*H*W,)`` indices, or ``(B, H*W)``-shaped indices.
        shape : (B, D, H, W) when channel_first=True (the convention used by
            train_gear/ar's ``qzshape``); ignored when ``indices`` already
            carries the spatial axes.
        channel_first : if True, the returned tensor is (B, D, H, W).
        """
        # Flatten to (N, D), then reshape per the requested layout.
        flat = indices.reshape(-1)
        codes = self.indices_to_codes(flat)  # (N, D)

        if shape is not None:
            if channel_first:
                b, _d, h, w = shape  # _d == self.codebook_dim implicitly
                # codes is (B*H*W, D) -> (B, H, W, D) -> (B, D, H, W)
                codes = codes.view(b, h, w, self.codebook_dim).permute(0, 3, 1, 2).contiguous()
            else:
                codes = codes.view(*shape)
        return codes

    # ---- forward (matches VectorQuantizer's contract) ---------------------
    def forward(self, z: torch.Tensor, return_distance: bool = False):
        """Quantize ``z`` of shape ``(B, D, H, W)``.

        Returns ``(quantized, (vq_l, commit_l, ent_l, usage), info)`` where
        ``info = (None, None, indices_flat)`` and ``ent_l`` is the 3-tuple
        ``(scaled_total, sample_entropy, codebook_entropy)`` -- matching
        ``models/vq_model.VectorQuantizer`` so the outer training code does
        not need to branch on quantizer type.

        ``return_distance=True`` additionally returns a ``(B*H*W,
        codebook_size)`` tensor ``d`` such that ``softmax(-d/T)`` is the
        standard LFQ soft assignment over the full hypercube codebook.
        Functionally analogous to VQ's squared distance -- we use
        ``d = -2 * <z, codebook>`` so the *sign* convention matches.
        """
        b, _d, h, w = z.shape
        # (B, D, H, W) -> (B, H*W, D)
        x = z.permute(0, 2, 3, 1).contiguous().view(b, h * w, self.codebook_dim)

        # Sign quantization. Use ``> 0`` (zeros map to -1) to match the
        # reference implementation byte-for-byte.
        codebook_value = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        quantized = torch.where(x > 0, codebook_value, -codebook_value)
        # bit-pack -> integer indices in [0, K)
        indices_bn = ((quantized > 0).int() * self.mask.int()).sum(dim=-1)  # (B, H*W)

        # ``logits = 2 * <x, codebook>`` are the un-temperature LFQ
        # affinities. We compute them at most once and reuse them for
        # both the entropy aux loss (training) and the distance tensor
        # (``return_distance=True``). They MUST keep gradients flowing
        # to ``x`` so the REPA-E path in train_gear.py
        # (``soft = softmax(-d / T)`` -> AR -> proj_loss_vq -> encoder)
        # can update the encoder through the soft posterior.
        need_logits = self.training or return_distance
        if need_logits:
            logits = 2.0 * torch.einsum("bnd,kd->bnk", x, self.codebook)
        else:
            logits = None

        # entropy aux loss + commit MSE (training only)
        if self.training:
            per_sample_entropy, codebook_entropy, entropy_aux_loss = _entropy_loss_from_logits(
                logits,
                temperature=self.entropy_loss_temperature,
                sample_minimization_weight=self.sample_minimization_weight,
                batch_maximization_weight=self.batch_maximization_weight,
                cross_rank_avg_entropy=self.training,
            )
            # Pre-multiply by ``commit_loss_beta`` (default 0.25) to match
            # ``VectorQuantizer.beta`` semantics: outer ``losses.py`` only
            # multiplies by ``quantizer_weight=1.0``, so the effective term
            # is ``commit_loss_beta * MSE(z_e, sg(z_q))``. With the default
            # this matches ``commit_weight=0.25`` from MAGVIT2's
            # ``pretrain_lfqgan_256_16384.yaml``.
            commit_loss = self.commit_loss_beta * F.mse_loss(x, quantized.detach())
        else:
            zero = torch.zeros((), device=x.device, dtype=x.dtype)
            per_sample_entropy = codebook_entropy = entropy_aux_loss = zero
            commit_loss = zero

        # Straight-through: forward = quantized, backward = identity through x.
        quantized = x + (quantized - x).detach()
        # (B, H*W, D) -> (B, D, H, W)
        quantized = quantized.view(b, h, w, self.codebook_dim).permute(0, 3, 1, 2).contiguous()

        # vq_loss is N/A for LFQ (no learnable codebook); set to None so the
        # outer loss helper coerces it to zero. commit_loss is the canonical
        # MSE between the encoder output and the bit-quantized version.
        vq_l = None
        commit_l = commit_loss
        ent_l = (
            self.entropy_loss_ratio * entropy_aux_loss,
            per_sample_entropy,
            codebook_entropy,
        )
        # Stage 0/1 compute global codebook_usage themselves via
        # ``accelerator.gather(indices)`` -> we keep the slot but pass 0.
        usage = 0

        # Flat indices for AR consumption: (B*H*W,). Stage 2 immediately
        # does ``indices.reshape(B, -1)`` to recover (B, H*W).
        indices_flat = indices_bn.reshape(-1)
        info = (None, None, indices_flat)

        if return_distance:
            # (B, H*W, K) -> (B*H*W, K). ``d = -logits`` so the standard
            # ``softmax(-d / T)`` consumed by train_gear / train_tokenizer
            # reduces to ``softmax(logits / T)`` -- the canonical LFQ soft
            # assignment over the full hypercube codebook.
            d = (-logits).reshape(-1, self.codebook_size)
            return quantized, (vq_l, commit_l, ent_l, usage), info, d
        return quantized, (vq_l, commit_l, ent_l, usage), info


def _entropy_loss_from_logits(
    logits: torch.Tensor,
    temperature: float,
    sample_minimization_weight: float,
    batch_maximization_weight: float,
    eps: float = 1e-5,
    cross_rank_avg_entropy: bool = True,
):
    """Same form as ``Open_MAGVIT2.modules.vqvae.lookup_free_quantize.
    entropy_loss``, but operating directly on a (B, N, K) logits tensor.

    Returns ``(sample_entropy, codebook_entropy, total)`` with the sign
    convention ``total = w_s * sample_entropy - w_b * codebook_entropy``.

    DDP correctness: ``codebook_entropy`` is a non-linear function of the
    batch-averaged distribution, so a naive per-rank computation biases
    the value (lower per-rank ceiling = log(N_local) instead of
    log(N_global)) and weakens the codebook-diversity pressure in
    multi-node runs. We do a one-shot ``all_reduce`` on the K-dim
    ``avg_probs`` vector and use a straight-through estimator so the
    autograd graph never sees the distributed op -- see
    ``models.vq_model.compute_entropy_loss`` for the full derivation.
    The ``cross_rank_avg_entropy`` gate keeps the function deadlock-safe
    in code paths where ranks may diverge -- the caller should pass
    ``cross_rank_avg_entropy=self.training``.
    """
    flat = logits.reshape(-1, logits.shape[-1])
    probs = F.softmax(flat / temperature, dim=-1)
    log_probs = F.log_softmax(flat / temperature + eps, dim=-1)

    avg_probs_local = probs.mean(dim=0)
    if (
        cross_rank_avg_entropy
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        with torch.no_grad():
            avg_probs_global = avg_probs_local.detach().clone()
            dist.all_reduce(avg_probs_global, op=dist.ReduceOp.SUM)
            avg_probs_global = avg_probs_global / dist.get_world_size()
        # Straight-through: forward = global avg distribution, backward
        # grad flows through avg_probs_local. After DDP averages
        # gradients across world_size ranks the magnitude comes out
        # exact. See ``models.vq_model.compute_entropy_loss``.
        avg_probs = avg_probs_local + (avg_probs_global - avg_probs_local).detach()
    else:
        avg_probs = avg_probs_local
    codebook_entropy = -(avg_probs * torch.log(avg_probs + eps)).sum()

    sample_entropy = -(probs * log_probs).sum(dim=-1).mean()

    total = (
        sample_minimization_weight * sample_entropy
        - batch_maximization_weight * codebook_entropy
    )
    return sample_entropy, codebook_entropy, total


# =============================================================================
# Top-level LFQ model -- the drop-in for ``models.vq_model.VQModel``
# =============================================================================
@dataclass
class LFQModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 14  # must equal log2(codebook_size)
    z_channels: int = 14          # tied to codebook_embed_dim for single-codebook LFQ

    ch: int = 128
    ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    num_res_blocks: int = 4
    in_channels: int = 3
    out_ch: int = 3
    resolution: int = 256

    sample_minimization_weight: float = 1.0
    batch_maximization_weight: float = 1.0
    entropy_loss_ratio: float = 1.0
    entropy_loss_temperature: float = 0.01
    # Pre-multiplier on the commit MSE returned by the quantizer; mirrors
    # ``ModelArgs.commit_loss_beta`` so VQ and LFQ have the same outer-loss
    # contract. Default 0.25 matches MAGVIT2's ``commit_weight: 0.25`` and
    # the LlamaGen VQ-VAE default. Bumping this up to 1.0 reproduces the
    # 4x-too-strong-commit regime that, on a warm-started LFQ ckpt, shows
    # up as the ``commit_loss`` curve spiking to ~0.4 then collapsing.
    commit_loss_beta: float = 0.25

    # For naming symmetry with ModelArgs in vq_model.py -- not used by LFQ.
    dropout_p: float = 0.0


class LFQModel(nn.Module):
    """Open-MAGVIT2-style LFQ tokenizer with the ``models.vq_model.VQModel`` API.

    Differences vs. VQModel:
      * No ``quant_conv`` / ``post_quant_conv`` (LFQ keeps z_channels == codebook_dim,
        so the encoder/decoder feed the quantizer directly).
      * ``quantize`` is an LFQ hypercube quantizer, not a learnable codebook.
      * The forward returns the same outer tuple shape as VQModel so the
        Stage 0/1 training loop (`vq(..., return_distance=True)`) and the
        Stage 2 inference path (`vq.encode(...)`, `vq.decode_code(...)`)
        consume the result without branching on type.
    """

    def __init__(self, config: LFQModelArgs):
        super().__init__()
        # Resolve / validate embed dim. Single-codebook LFQ forces
        # embed_dim == log2(codebook_size); fall back to that if the
        # caller passed something falsy (e.g. the VQ default of 8).
        cb_dim = int(log2(config.codebook_size))
        if not config.codebook_embed_dim or config.codebook_embed_dim != cb_dim:
            config.codebook_embed_dim = cb_dim
            config.z_channels = cb_dim
        if not config.z_channels:
            config.z_channels = config.codebook_embed_dim

        self.config = config

        self.encoder = Encoder(
            ch=config.ch,
            out_ch=config.out_ch,
            in_channels=config.in_channels,
            num_res_blocks=config.num_res_blocks,
            z_channels=config.z_channels,
            ch_mult=tuple(config.ch_mult),
            resolution=config.resolution,
        )
        self.decoder = Decoder(
            ch=config.ch,
            out_ch=config.out_ch,
            in_channels=config.in_channels,
            num_res_blocks=config.num_res_blocks,
            z_channels=config.z_channels,
            ch_mult=tuple(config.ch_mult),
            resolution=config.resolution,
        )
        self.quantize = LFQQuantizer(
            codebook_size=config.codebook_size,
            embed_dim=config.codebook_embed_dim,
            sample_minimization_weight=config.sample_minimization_weight,
            batch_maximization_weight=config.batch_maximization_weight,
            entropy_loss_ratio=config.entropy_loss_ratio,
            entropy_loss_temperature=config.entropy_loss_temperature,
            commit_loss_beta=config.commit_loss_beta,
        )

    # ---- VQModel API ------------------------------------------------------
    def encode(self, x: torch.Tensor, return_distance: bool = False):
        h = self.encoder(x)
        out = self.quantize(h, return_distance=return_distance)
        if return_distance:
            quant, diff, info, d = out
            return quant, diff, info, d
        quant, diff, info = out
        return quant, diff, info

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        return self.decoder(quant)

    def decode_code(
        self,
        code_b: torch.Tensor,
        shape: Optional[Tuple[int, int, int, int]] = None,
        channel_first: bool = True,
    ) -> torch.Tensor:
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        return self.decode(quant_b)

    def forward(self, input: torch.Tensor, return_distance: bool = False):
        if return_distance:
            quant, diff, info, d = self.encode(input, return_distance=True)
            dec = self.decode(quant)
            return dec, quant, diff, info, d
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff


# =============================================================================
# TencentARC Open-MAGVIT2 pretrained-checkpoint converter
# =============================================================================
def convert_magvit2_pretrained_state_dict(
    state_dict: "OrderedDict[str, torch.Tensor]",
    prefer_ema: bool = True,
) -> "OrderedDict[str, torch.Tensor]":
    """Translate a TencentARC Open-MAGVIT2 Lightning ``state_dict`` into
    one that ``LFQModel.load_state_dict(..., strict=False)`` accepts.

    The MAGVIT2 ckpt contains:

      * ``encoder.*`` / ``decoder.*``         -- live params
      * ``model_ema.decay``                    -- EMA decay (skipped)
      * ``model_ema.<key with dots stripped>`` -- EMA shadows of the live
        encoder/decoder params (LitEma stores names without ``.`` because
        they would otherwise be parsed as submodule attributes)
      * (optional) ``loss.*`` / ``inception*`` / ``lpips*``  -- training-time
        helpers that have no analogue in our LFQModel

    Strategy:
      1. Walk the live ``encoder.*`` / ``decoder.*`` keys; build a map
         ``stripped_name -> original_name`` so the EMA shadows can be
         reconnected to their parent param.
      2. If ``prefer_ema`` and the EMA shadow exists, return it under
         the live param name; otherwise return the live param.
      3. Drop everything else.

    Parameters
    ----------
    state_dict : OrderedDict[str, Tensor]
        The raw lightning state_dict (i.e. ``ckpt['state_dict']``).
    prefer_ema : bool
        Whether to prefer the EMA copy over the live params when both are
        present. The MAGVIT2 inference pipeline uses EMA by default
        (see lfqgan_pretrain.VQModel.init_from_ckpt's ``stage='transformer'``
        branch); we match that default.

    Returns
    -------
    OrderedDict[str, Tensor]
        Keys are bare ``encoder.*`` / ``decoder.*`` paths matching the
        ``LFQModel`` submodule layout (no ``model_ema.`` prefix).
    """
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    stripped_to_real: dict = {}

    for k, v in state_dict.items():
        if k.startswith("model_ema."):
            continue
        if k.startswith("encoder.") or k.startswith("decoder."):
            out[k] = v
            stripped_to_real[k.replace(".", "")] = k

    if not prefer_ema:
        return out

    for k, v in state_dict.items():
        if not k.startswith("model_ema."):
            continue
        sub = k[len("model_ema."):]
        if sub == "decay" or sub == "num_updates":
            continue
        real = stripped_to_real.get(sub)
        if real is None:
            # EMA shadow without a matching live param -- silently drop.
            continue
        out[real] = v

    return out


def load_magvit2_pretrained_(
    model: LFQModel,
    ckpt_path: str,
    prefer_ema: bool = True,
    map_location: str = "cpu",
):
    """Convenience helper: load a MAGVIT2 ``.ckpt`` directly into an
    instantiated :class:`LFQModel`. Returns ``(missing, unexpected)`` from
    ``load_state_dict``.
    """
    raw = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    sd = raw["state_dict"] if "state_dict" in raw else raw
    converted = convert_magvit2_pretrained_state_dict(sd, prefer_ema=prefer_ema)
    return model.load_state_dict(converted, strict=False)


# =============================================================================
# LFQ Model factory + registry (mirrors VQ_models in models/vq_model.py)
# =============================================================================
def LFQ_16(**kwargs):
    """LFQ tokenizer, downsample ratio 16, ch=128, 4 res blocks per level.

    Mirrors the TencentARC ``pretrain_lfqgan_256_16384.yaml`` architecture:
    ``ch=128, ch_mult=[1,1,2,2,4], num_res_blocks=4, z_channels=14``. The
    ``codebook_size`` / ``codebook_embed_dim`` kwargs are forwarded so the
    Stage 0/1/2 scripts can keep their ``--codebook-size`` / ``--codebook-embed-dim``
    flags exactly as for VQ.
    """
    # Default to the published MAGVIT2-16384 shape; let caller override.
    args = dict(
        codebook_size=16384,
        codebook_embed_dim=14,
        z_channels=14,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=4,
    )
    args.update(kwargs)
    return LFQModel(LFQModelArgs(**args))


LFQ_models = {"LFQ-16": LFQ_16}
