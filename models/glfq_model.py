"""Grouped / variable-downsample Lookup-Free Quantization (GLFQ) tokenizer.

This is a *self-contained* sibling of ``models/lfq_model.py``. It is kept
in its own file so the original ``LFQ-16`` recipe (and every script that
already uses it) is byte-for-byte unaffected.

Two things differ from ``models/lfq_model.py``:

1. **Variable downsample with a shared backbone.** The encoder/decoder
   reuse the exact MAGVIT2 backbone (``ch=128, ch_mult=[1,1,2,2,4],
   num_res_blocks=4``) but can *skip* a configurable number of
   down/up-sample layers while keeping all the ResBlocks. Concretely an
   8x model is just the 16x model with the **last encoder downsample**
   and the **first decoder upsample** removed, so the two models share
   essentially the same learnable parameters (they differ only by the one
   dropped downsample conv / one dropped upsampler, plus the
   ``z_channels``-dependent ``conv_in`` / ``conv_out`` / AdaptiveGroupNorm
   projections). This is exactly the "keep the learnable params basically
   identical" setup requested for the dim8-8x vs dim32-16x comparison.

2. **Grouped entropy aux loss.** A single-codebook LFQ with ``dim=D`` has
   a ``2**D`` hypercube codebook. For ``D=32`` that is ``2**32`` codes,
   which cannot be materialized for the entropy auxiliary loss. The
   ``GLFQQuantizer`` keeps the quantization itself purely per-dimension
   (``sign(z) -> {-1,+1}`` over all ``D`` dims, numerically identical to
   ungrouped LFQ), but computes the *entropy aux loss* by splitting the
   ``D`` dims into ``num_codebooks`` groups of ``D/num_codebooks`` bits
   each, materializing only the small ``2**(D/num_codebooks)`` per-group
   codebook, computing the LFQ entropy loss within each group, and
   **averaging the per-group losses**. With ``num_codebooks=1`` this
   reduces to the standard single-codebook LFQ entropy loss, so
   ``dim=8`` (``2**8=256`` codes, easily materialized) can run with
   ``num_codebooks=1`` and stays identical to the reference recipe.

   Reference: GFQ / Open-MAGVIT2 lookup-free quantization
   (https://arxiv.org/abs/2310.05737).

API contract is identical to ``models/lfq_model.py:LFQModel`` and
``models/vq_model.py:VQModel`` so ``src/train_tokenizer.py`` consumes it
through ``--vq-model`` without any branching:

* ``model.encode(x[, return_distance=True])`` -> ``(quant, diff, info[, d])``
* ``model.decode(quant)`` -> recon
* ``model.decode_code(idx, shape)`` -> recon
* ``model(x[, return_distance=True])`` -> ``(recon, diff)`` /
  ``(recon, quant, diff, info, d)``

where ``diff = (vq_loss, commit_loss, entropy_loss_tuple, usage)`` and
``info = (None, None, indices_flat)``.

Diagnostics note: because the full ``2**D`` codebook is never
materialized for ``D>~20``, the ``return_distance=True`` distance tensor
``d`` and the ``info`` indices are reported **per group** (over the small
``2**(D/num_codebooks)`` sub-codebook). The Stage-0 ``codebook/*`` /
``codebook_usage`` diagnostics in ``train_tokenizer.py`` therefore track
the per-group sub-codebook statistics. The *reconstruction* path (the
quantized ``{-1,+1}^D`` latent) is unaffected and exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import log2
from typing import List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the (unchanged) building blocks + DDP-safe entropy helper from the
# original LFQ port so the two stay in sync and the backbone is provably
# identical.
from .lfq_model import (
    AdaptiveGroupNorm,
    ResBlock,
    Upsampler,
    _entropy_loss_from_logits,
    swish,
)


# =============================================================================
# Encoder / Decoder with selectable down/up-sample levels.
#
# Structurally identical to ``lfq_model.Encoder`` / ``Decoder`` (same
# ResBlock stack, same AdaptiveGroupNorm, same channel flow) EXCEPT the set
# of levels that actually carry a downsample conv (encoder) / upsampler
# (decoder) is configurable. Skipping a down/up-sample layer does not change
# the channel flow (the downsample conv is block_out->block_out and the
# Upsampler is block_in->block_in via depth-to-space), so every ResBlock
# keeps the exact same shape as in the 16x model.
# =============================================================================
class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        in_channels: int,
        num_res_blocks: int,
        z_channels: int,
        ch_mult: Tuple[int, ...] = (1, 1, 2, 2, 4),
        resolution: int = 256,
        downsample_levels: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        del out_ch, resolution
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(ch_mult)

        # Which i_level indices carry a stride-2 downsample. Default mirrors
        # the original Encoder (every non-final level).
        if downsample_levels is None:
            downsample_levels = list(range(self.num_blocks - 1))
        self.downsample_levels: Set[int] = set(int(i) for i in downsample_levels)

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
            if i_level in self.downsample_levels:
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
            if i_level in self.downsample_levels:
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
        ch_mult: Tuple[int, ...] = (1, 1, 2, 2, 4),
        resolution: int = 256,
        upsample_levels: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        del in_channels, resolution
        self.ch = ch
        self.num_blocks = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # Which i_level indices carry an upsampler. Default mirrors the
        # original Decoder (every level with i_level > 0).
        if upsample_levels is None:
            upsample_levels = list(range(1, self.num_blocks))
        self.upsample_levels: Set[int] = set(int(i) for i in upsample_levels)

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
            if i_level in self.upsample_levels:
                up.upsample = Upsampler(block_in)
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, padding=1)

    @property
    def last_layer(self):
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
            if i_level in self.upsample_levels:
                z = self.up[i_level].upsample(z)
        z = self.norm_out(z)
        z = swish(z)
        z = self.conv_out(z)
        return z


# =============================================================================
# Grouped Lookup-Free Quantizer.
# =============================================================================
class GLFQQuantizer(nn.Module):
    """Lookup-free quantizer with a *grouped* entropy auxiliary loss.

    The quantization is per-dimension ``sign(z) -> {-1,+1}`` over all
    ``dim`` channels (a straight-through estimator), exactly like the
    single-codebook LFQ. The only thing the grouping changes is *how the
    entropy aux loss is computed*: the ``dim`` channels are split into
    ``num_codebooks`` contiguous groups, the LFQ entropy loss is computed
    within each group against the small ``2**(dim//num_codebooks)``
    per-group hypercube codebook, and the per-group losses are averaged.

    ``num_codebooks == 1`` reproduces the standard single-codebook LFQ
    entropy loss exactly.
    """

    def __init__(
        self,
        dim: int,
        num_codebooks: int = 1,
        sample_minimization_weight: float = 1.0,
        batch_maximization_weight: float = 1.0,
        entropy_loss_ratio: float = 1.0,
        entropy_loss_temperature: float = 0.01,
        commit_loss_beta: float = 0.25,
    ):
        super().__init__()
        assert dim % num_codebooks == 0, (
            f"dim ({dim}) must be divisible by num_codebooks ({num_codebooks})."
        )
        self.dim = int(dim)
        self.num_codebooks = int(num_codebooks)
        self.group_dim = self.dim // self.num_codebooks
        # Per-group sub-codebook size (the size actually materialized).
        self.per_group_size = 2 ** self.group_dim
        # The "true" lookup-free codebook size (never materialized for big dim).
        self.codebook_size = 2 ** self.dim

        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight
        self.entropy_loss_ratio = float(entropy_loss_ratio)
        self.entropy_loss_temperature = float(entropy_loss_temperature)
        self.commit_loss_beta = float(commit_loss_beta)

        # Bit mask for per-group index <-> bit conversions (length group_dim).
        self.register_buffer(
            "mask", 2 ** torch.arange(self.group_dim), persistent=False,
        )
        # Pre-materialized {-1,+1}^group_dim per-group codebook for entropy.
        all_codes = torch.arange(self.per_group_size)
        bits = self._indices_to_bits(all_codes, self.group_dim)
        codebook = bits.float() * 2.0 - 1.0  # (per_group_size, group_dim)
        self.register_buffer("codebook", codebook, persistent=False)

    # ---- bit / index conversions ------------------------------------------
    @staticmethod
    def _indices_to_bits(x: torch.Tensor, n_bits: int) -> torch.Tensor:
        mask = 2 ** torch.arange(n_bits, device=x.device, dtype=torch.long)
        return (x.unsqueeze(-1) & mask) != 0

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Per-group indices (values in ``[0, per_group_size)``) -> ``{-1,+1}``
        vectors with a trailing dim of size ``group_dim``.
        """
        bits = self._indices_to_bits(indices, self.group_dim)
        return bits.to(self.codebook.dtype) * 2.0 - 1.0

    def get_codebook_entry(
        self,
        indices: torch.Tensor,
        shape: Optional[Tuple[int, int, int, int]] = None,
        channel_first: bool = True,
    ) -> torch.Tensor:
        """Rebuild the ``{-1,+1}^dim`` latent from **per-group** indices.

        ``indices`` is expected to carry the per-group axis as its last dim
        (``(..., num_codebooks)``) or be flat with a length divisible by
        ``num_codebooks``. ``shape = (B, dim, H, W)`` when
        ``channel_first`` (the convention used by train_gear/ar's
        ``qzshape``).
        """
        flat = indices.reshape(-1, self.num_codebooks)  # (B*H*W, G)
        codes = self.indices_to_codes(flat)             # (B*H*W, G, group_dim)
        codes = codes.reshape(flat.shape[0], self.dim)  # (B*H*W, dim)

        if shape is not None:
            if channel_first:
                b, _d, h, w = shape
                codes = codes.view(b, h, w, self.dim).permute(0, 3, 1, 2).contiguous()
            else:
                codes = codes.view(*shape)
        return codes

    # ---- forward (matches VectorQuantizer / LFQQuantizer contract) --------
    def forward(self, z: torch.Tensor, return_distance: bool = False):
        b, dim, h, w = z.shape
        assert dim == self.dim, f"expected {self.dim} channels, got {dim}"
        n = h * w
        g = self.group_dim
        G = self.num_codebooks

        # (B, D, H, W) -> (B, N, D)
        x = z.permute(0, 2, 3, 1).contiguous().view(b, n, self.dim)

        codebook_value = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        quantized = torch.where(x > 0, codebook_value, -codebook_value)  # (B, N, D)

        # Grouped views for entropy / indices / distance.
        xg = x.view(b, n, G, g)                     # (B, N, G, g)
        qg = quantized.view(b, n, G, g)             # (B, N, G, g)
        indices_g = ((qg > 0).int() * self.mask.int()).sum(dim=-1)  # (B, N, G)

        need_logits = self.training or return_distance
        if need_logits:
            # (B, N, G, g) x (per_group_size, g) -> (B, N, G, per_group_size)
            logits = 2.0 * torch.einsum("bngd,kd->bngk", xg, self.codebook)
        else:
            logits = None

        if self.training:
            # Per-group entropy loss, then averaged over the G groups. With
            # G==1 this is identical to the single-codebook LFQ entropy loss.
            sample_acc = x.new_zeros(())
            codebook_acc = x.new_zeros(())
            total_acc = x.new_zeros(())
            for gi in range(G):
                se, ce, tot = _entropy_loss_from_logits(
                    logits[:, :, gi, :],  # (B, N, per_group_size)
                    temperature=self.entropy_loss_temperature,
                    sample_minimization_weight=self.sample_minimization_weight,
                    batch_maximization_weight=self.batch_maximization_weight,
                    cross_rank_avg_entropy=self.training,
                )
                sample_acc = sample_acc + se
                codebook_acc = codebook_acc + ce
                total_acc = total_acc + tot
            per_sample_entropy = sample_acc / G
            codebook_entropy = codebook_acc / G
            entropy_aux_loss = total_acc / G

            commit_loss = self.commit_loss_beta * F.mse_loss(x, quantized.detach())
        else:
            zero = torch.zeros((), device=x.device, dtype=x.dtype)
            per_sample_entropy = codebook_entropy = entropy_aux_loss = zero
            commit_loss = zero

        # Straight-through: forward = quantized, backward = identity through x.
        quantized = x + (quantized - x).detach()
        quantized = quantized.view(b, h, w, self.dim).permute(0, 3, 1, 2).contiguous()

        vq_l = None
        commit_l = commit_loss
        ent_l = (
            self.entropy_loss_ratio * entropy_aux_loss,
            per_sample_entropy,
            codebook_entropy,
        )
        usage = 0

        # Per-group flat indices (over the per_group_size sub-codebook).
        indices_flat = indices_g.reshape(-1)
        info = (None, None, indices_flat)

        if return_distance:
            # (B, N, G, per_group_size) -> (B*N*G, per_group_size).
            d = (-logits).reshape(-1, self.per_group_size)
            return quantized, (vq_l, commit_l, ent_l, usage), info, d
        return quantized, (vq_l, commit_l, ent_l, usage), info


# =============================================================================
# Top-level GLFQ model -- drop-in for ``models.vq_model.VQModel``.
# =============================================================================
@dataclass
class GLFQModelArgs:
    # ``dim`` is the per-token bit dimension (== log2 of the lookup-free
    # codebook size). Set via ``codebook_embed_dim`` from the CLI.
    codebook_embed_dim: int = 32
    num_codebooks: int = 4

    # Spatial downsample ratio (8 or 16). The backbone is shared; the model
    # skips the trailing encoder downsamples / leading decoder upsamples to
    # realize ``downsample`` while keeping the ResBlock stack intact.
    downsample: int = 16

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
    commit_loss_beta: float = 0.25

    # For naming symmetry with ModelArgs in vq_model.py -- not used by GLFQ.
    dropout_p: float = 0.0


class GLFQModel(nn.Module):
    """Open-MAGVIT2-style LFQ tokenizer with selectable downsample + grouped
    entropy loss. API mirrors ``models.lfq_model.LFQModel``.
    """

    def __init__(self, config: GLFQModelArgs):
        super().__init__()
        dim = int(config.codebook_embed_dim)
        assert dim >= 1, f"codebook_embed_dim (dim) must be >= 1, got {dim}."
        # z_channels is tied to the bit dim for lookup-free quantization.
        z_channels = dim

        num_blocks = len(config.ch_mult)
        max_down = num_blocks - 1  # backbone supports up to this many down/up samples
        needed = int(round(log2(config.downsample)))
        assert 2 ** needed == config.downsample, (
            f"downsample must be a power of 2 (got {config.downsample})."
        )
        assert 1 <= needed <= max_down, (
            f"downsample={config.downsample} needs {needed} down/up samples, but the "
            f"backbone ch_mult={config.ch_mult} supports at most {max_down}."
        )
        num_skip = max_down - needed

        # Encoder: keep the FIRST ``needed`` downsample levels (i.e. drop the
        # LAST ``num_skip`` downsamples). Decoder: keep the upsamplers at
        # i_level in [1 .. num_blocks-1-num_skip] (i.e. drop the FIRST
        # ``num_skip`` upsamplers encountered in the reversed forward pass,
        # which are the highest i_level ones).
        downsample_levels = list(range(needed))
        upsample_levels = list(range(1, num_blocks - num_skip))

        self.config = config
        self.dim = dim
        self.num_codebooks = int(config.num_codebooks)
        self.downsample = int(config.downsample)

        self.encoder = Encoder(
            ch=config.ch,
            out_ch=config.out_ch,
            in_channels=config.in_channels,
            num_res_blocks=config.num_res_blocks,
            z_channels=z_channels,
            ch_mult=tuple(config.ch_mult),
            resolution=config.resolution,
            downsample_levels=downsample_levels,
        )
        self.decoder = Decoder(
            ch=config.ch,
            out_ch=config.out_ch,
            in_channels=config.in_channels,
            num_res_blocks=config.num_res_blocks,
            z_channels=z_channels,
            ch_mult=tuple(config.ch_mult),
            resolution=config.resolution,
            upsample_levels=upsample_levels,
        )
        self.quantize = GLFQQuantizer(
            dim=dim,
            num_codebooks=config.num_codebooks,
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
# Factories + registry (additive; mirrors LFQ_models in models/lfq_model.py)
# =============================================================================
def _build_glfq(default_dim: int, default_groups: int, default_downsample: int, **kwargs):
    """Shared factory body.

    ``train_tokenizer.py`` always passes ``codebook_size`` and
    ``codebook_embed_dim``. For lookup-free quantization the codebook size
    is fully determined by the bit dim (``2**dim``), so ``codebook_size``
    is validated-then-dropped here (it is not a constructor field). The bit
    dim comes from ``codebook_embed_dim``. ``num_codebooks`` may be
    overridden via the optional ``--entropy-num-groups`` CLI flag.
    """
    dim = int(kwargs.pop("codebook_embed_dim", default_dim) or default_dim)

    codebook_size = kwargs.pop("codebook_size", None)
    if codebook_size is not None:
        assert int(codebook_size) == 2 ** dim, (
            f"For lookup-free quantization codebook_size must equal 2**dim. "
            f"Got codebook_size={codebook_size}, dim={dim} (2**dim={2 ** dim})."
        )

    args = dict(
        codebook_embed_dim=dim,
        num_codebooks=default_groups,
        downsample=default_downsample,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=4,
    )
    args.update(kwargs)
    return GLFQModel(GLFQModelArgs(**args))


def LFQ_8(**kwargs):
    """8x-downsample lookup-free tokenizer, dim=8 (codebook 2**8=256).

    Same MAGVIT2 backbone as ``LFQ-16`` but with the last encoder
    downsample + first decoder upsample removed (-> 8x instead of 16x),
    so the learnable params match the 16x model almost exactly. With a
    256-code hypercube the entropy loss needs no grouping
    (``num_codebooks=1`` == standard single-codebook LFQ entropy).
    """
    return _build_glfq(default_dim=8, default_groups=1, default_downsample=8, **kwargs)


def GLFQ_16(**kwargs):
    """16x-downsample lookup-free tokenizer, dim=32 (codebook 2**32).

    Same MAGVIT2 backbone as ``LFQ-16``. The 2**32 hypercube cannot be
    materialized for the entropy loss, so the entropy aux loss is computed
    per group (``num_codebooks=4`` -> 4 groups of 8 bits / 256 codes each)
    and averaged. The quantization itself stays per-dimension {-1,+1} over
    all 32 dims (numerically identical to ungrouped LFQ).
    """
    return _build_glfq(default_dim=32, default_groups=4, default_downsample=16, **kwargs)


GLFQ_models = {"LFQ-8": LFQ_8, "GLFQ-16": GLFQ_16}
