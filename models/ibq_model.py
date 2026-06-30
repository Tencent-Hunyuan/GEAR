"""IBQ (Index-Backpropagation Quantization) tokenizer ported to plain PyTorch.

The reference implementation lives at
``SEED-Voken/src/IBQ/models/ibqgan.py`` and
``SEED-Voken/src/IBQ/modules/{diffusionmodules/model.py,vqvae/quantize.py}``
and is written for PyTorch Lightning + LitEma. This module strips Lightning
out and exposes an ``nn.Module`` whose API mirrors
``models/vq_model.VQModel`` (and ``models/lfq_model.LFQModel``) so it drops
into the same Stage 0/1/2 training scripts through a ``--vq-model`` flag.

API contract (matches ``models.vq_model.VQModel`` so train_tokenizer/gear/ar can
swap one for the other):

* ``IBQ_16(codebook_size, codebook_embed_dim, ...) -> IBQModel``
* ``model.encode(x)``                 -> ``(quant, diff, info)``
* ``model.encode(x, return_distance=True)``
                                      -> ``(quant, diff, info, d)``
* ``model.decode(quant)``             -> recon
* ``model.decode_code(idx, shape)``   -> recon (shape = (B, D, H, W))
* ``model(input)``                    -> ``(recon, diff)``
* ``model(input, return_distance=True)``
                                      -> ``(recon, quant, diff, info, d)``

where ``diff = (vq_loss, commit_loss, entropy_loss_tuple, codebook_usage)``
and ``info = (perp_or_None, min_encodings_or_None, min_encoding_indices)`` --
the slot shapes / semantics match ``VectorQuantizer`` so the outer
``ReconstructionLossVQ`` / train loops are byte-compatible.

IBQ specifics
-------------
IBQ replaces the L2 nearest-neighbour lookup with a *softmax over inner
products* against a learnable codebook, then uses a straight-through hard
one-hot for the forward pass while letting the gradient flow through the
soft assignment (the "index backpropagation" trick). Concretely, with
``logits = <z, e_k>`` the soft posterior is ``softmax(logits)`` over the
codebook. We therefore expose ``return_distance=True`` as
``d = -logits`` so the standard ``softmax(-d / T)`` consumed by
``src/train_gear.py`` reduces to the IBQ soft assignment
``softmax(logits / T)``.

Pretrained-weight loading: ``convert_ibq_pretrained_state_dict`` takes a
TencentARC IBQ Lightning ``state_dict`` (which carries both live
``encoder.*`` / ``decoder.*`` / ``quant_conv.*`` / ``post_quant_conv.*`` /
``quantize.embedding.*`` params and ``model_ema.<key with dots stripped>``
LitEma shadows) and returns a state_dict that
``IBQModel.load_state_dict(..., strict=False)`` accepts.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import einsum


# =============================================================================
# Encoder / Decoder (ported verbatim from
# SEED-Voken/src/IBQ/modules/diffusionmodules/model.py -- the taming /
# LDM autoencoder). Kept structurally identical so the TencentARC IBQ
# pretrained state_dict drops in. IBQ uses ``temb_channels=0`` everywhere,
# so the timestep-embedding path of the reference ResnetBlock is never
# constructed.
# =============================================================================
def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels,
                                        kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        # Match VQ_model.Decoder API used by `ReconstructionLossVQ`'s
        # adaptive disc-weight path.
        return self.conv_out.weight

    def forward(self, z):
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


# =============================================================================
# Index-Backpropagation Quantizer (adapted from
# SEED-Voken's quantize.IndexPropagationQuantize).
#
# Differences vs. the reference:
#   * Return tuple shaped to match models/vq_model.py:VectorQuantizer so the
#     outer ReconstructionLossVQ and train loops don't need to special-case.
#     ``diff = (vq_loss, commit_loss, entropy_loss_tuple, codebook_usage)``;
#     IBQ folds its full quant loss (the soft+hard reconstruction + commit
#     term) into the ``vq_loss`` slot and leaves ``commit_loss=None`` so the
#     outer ``vq_loss + commit_loss`` sum reproduces the IBQ recipe exactly.
#   * ``return_distance=True`` exposes a (B*H*W, K) "distance" tensor
#     ``d = -logits`` whose ``softmax(-d / T)`` gives the IBQ soft
#     assignment over the codebook -- this is what src/train_gear.py
#     consumes for soft-label AR training.
# =============================================================================
def compute_entropy_loss(
    logits,
    temperature=0.01,
    sample_minimization_weight=1.0,
    batch_maximization_weight=1.0,
    eps=1e-5,
    cross_rank_avg_entropy: bool = True,
):
    """Entropy loss of unnormalized logits (affinities over the last dim).

    Ported verbatim from ``SEED-Voken/src/IBQ/modules/vqvae/quantize.py`` with
    one DDP-correctness fix: ``avg_entropy`` is computed from the
    *globally* averaged distribution (one all_reduce on the K-dim probs
    vector) so its value -- and the resulting codebook-diversity pressure
    -- no longer depend on world size. See the docstring of
    ``models.vq_model.compute_entropy_loss`` for the full derivation
    (straight-through estimator, DDP gradient cancellation,
    torch.compile safety, etc.). The ``cross_rank_avg_entropy`` gate
    keeps the function deadlock-safe in code paths where ranks may
    diverge -- the caller should pass ``cross_rank_avg_entropy=self.training``.
    """
    probs = F.softmax(logits / temperature, -1)
    log_probs = F.log_softmax(logits / temperature + eps, -1)

    avg_probs_local = reduce(probs, "... D -> D", "mean")
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
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))

    sample_entropy = -torch.sum(probs * log_probs, -1)
    sample_entropy = torch.mean(sample_entropy)

    loss = (sample_minimization_weight * sample_entropy) - (
        batch_maximization_weight * avg_entropy
    )
    return sample_entropy, avg_entropy, loss


class IBQQuantizer(nn.Module):
    """Index-backpropagation quantizer with the API of ``VectorQuantizer``.

    The codebook is a learnable ``nn.Embedding(n_e, e_dim)``. The forward
    computes:

      * ``logits``   = ``<z, e_k>`` (inner-product affinity over codebook).
      * ``indices``  = ``argmax_k logits`` (hard assignment).
      * ``quantized``= straight-through hard one-hot @ codebook, with the
        soft posterior carrying the gradient (index backpropagation).
      * IBQ quant loss + optional entropy auxiliary loss during training
        (zeros at eval).
    """

    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float = 0.25,
        use_entropy_loss: bool = True,
        entropy_temperature: float = 0.01,
        sample_minimization_weight: float = 1.0,
        batch_maximization_weight: float = 1.0,
        entropy_loss_ratio: float = 1.0,
        cosine_similarity: bool = False,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.use_entropy_loss = use_entropy_loss
        self.cosine_similarity = cosine_similarity
        self.entropy_temperature = float(entropy_temperature)
        self.sample_minimization_weight = float(sample_minimization_weight)
        self.batch_maximization_weight = float(batch_maximization_weight)
        # Mutable from outside (Stage 0/1 set it from the --entropy-loss-ratio
        # CLI flag, mirroring how VectorQuantizer.entropy_loss_ratio is
        # overridden). Scales the entropy aux loss reported to the outer loss.
        self.entropy_loss_ratio = float(entropy_loss_ratio)

        self.embedding = nn.Embedding(self.n_e, self.e_dim)

    def forward(self, z: torch.Tensor, return_distance: bool = False):
        """Quantize ``z`` of shape ``(B, D, H, W)``.

        Returns ``(z_q, (vq_l, commit_l, ent_l, usage), info)`` where
        ``info = (None, None, indices_flat)`` and ``ent_l`` is the 3-tuple
        ``(scaled_total, sample_entropy, avg_entropy)`` -- matching
        ``models/vq_model.VectorQuantizer`` so the outer training code does
        not need to branch on quantizer type.

        ``return_distance=True`` additionally returns a ``(B*H*W, n_e)``
        tensor ``d = -logits`` such that ``softmax(-d / T)`` is the IBQ
        soft assignment over the codebook.
        """
        # z: [b, d, h, w]; embedding.weight: [n, d]
        if self.cosine_similarity:
            z_norm = F.normalize(z, dim=1)
            emb_norm = F.normalize(self.embedding.weight, dim=1)
            logits = einsum("b d h w, n d -> b n h w", z_norm, emb_norm)
        else:
            logits = einsum("b d h w, n d -> b n h w", z, self.embedding.weight)

        soft_one_hot = F.softmax(logits, dim=1)

        dim = 1
        ind = soft_one_hot.max(dim, keepdim=True)[1]
        hard_one_hot = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, ind, 1.0)
        # Straight-through: forward = hard one-hot, backward = soft posterior.
        one_hot = hard_one_hot - soft_one_hot.detach() + soft_one_hot

        z_q = einsum("b n h w, n d -> b d h w", one_hot, self.embedding.weight)
        z_q_2 = einsum("b n h w, n d -> b d h w", hard_one_hot, self.embedding.weight)

        vq_loss = None
        commit_loss = None
        entropy_loss_tuple = None
        if self.training:
            quant_loss = (
                torch.mean((z_q - z) ** 2)
                + torch.mean((z_q_2.detach() - z) ** 2)
                + self.beta * torch.mean((z_q_2 - z.detach()) ** 2)
            )
            # IBQ folds the commit term into ``quant_loss``; expose it via the
            # ``vq_loss`` slot so the outer ``vq_loss + commit_loss`` sum is
            # exactly the IBQ quant loss (commit_loss stays None -> coerced 0).
            vq_loss = quant_loss
            if self.use_entropy_loss:
                sample_entropy, avg_entropy, entropy_loss = compute_entropy_loss(
                    logits=logits.permute(0, 2, 3, 1).reshape(-1, self.n_e),
                    temperature=self.entropy_temperature,
                    sample_minimization_weight=self.sample_minimization_weight,
                    batch_maximization_weight=self.batch_maximization_weight,
                    cross_rank_avg_entropy=self.training,
                )
                entropy_loss_tuple = (
                    self.entropy_loss_ratio * entropy_loss,
                    sample_entropy,
                    avg_entropy,
                )

        # Flat indices for AR consumption: (B*H*W,). Stage 2 immediately
        # does ``indices.reshape(B, -1)`` to recover (B, H*W).
        ind_flat = torch.flatten(ind)
        info = (None, None, ind_flat)
        # codebook_usage is computed globally by the train loops via
        # ``accelerator.gather(indices)`` -> keep the slot but pass 0.
        diff = (vq_loss, commit_loss, entropy_loss_tuple, 0)

        if return_distance:
            # (B, n_e, H, W) -> (B*H*W, n_e). ``d = -logits`` so the standard
            # ``softmax(-d / T)`` consumed by train_gear / train_tokenizer
            # reduces to ``softmax(logits / T)`` -- the IBQ soft assignment.
            d = (-logits).permute(0, 2, 3, 1).reshape(-1, self.n_e)
            return z_q, diff, info, d
        return z_q, diff, info

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        """Mirror ``VectorQuantizer.get_codebook_entry``.

        Parameters
        ----------
        indices : LongTensor
            Flat ``(B*H*W,)`` indices (or any shape that flattens to that).
        shape : (B, D, H, W) when channel_first=True (the convention used by
            train_gear/ar's ``qzshape``); ignored when None.
        channel_first : if True, the returned tensor is (B, D, H, W).
        """
        flat = indices.reshape(-1)
        z_q = self.embedding(flat)  # (N, e_dim)

        if shape is not None:
            if channel_first:
                b, _d, h, w = shape
                z_q = z_q.view(b, h, w, self.e_dim).permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q


# =============================================================================
# Top-level IBQ model -- the drop-in for ``models.vq_model.VQModel``
# =============================================================================
@dataclass
class IBQModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 256
    z_channels: int = 256

    ch: int = 128
    ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    num_res_blocks: int = 4
    attn_resolutions: List[int] = field(default_factory=lambda: [16])
    in_channels: int = 3
    out_ch: int = 3
    resolution: int = 256
    dropout_p: float = 0.0

    beta: float = 0.25
    use_entropy_loss: bool = True
    entropy_loss_temperature: float = 0.01
    sample_minimization_weight: float = 1.0
    batch_maximization_weight: float = 1.0
    entropy_loss_ratio: float = 1.0
    cosine_similarity: bool = False

    # Pre-multiplier on the commit term, exposed for naming symmetry with
    # ``ModelArgs.commit_loss_beta`` (VQ) / ``LFQModelArgs.commit_loss_beta``
    # (LFQ). For IBQ this maps onto the quantizer's ``beta``.
    commit_loss_beta: float = 0.25


class IBQModel(nn.Module):
    """IBQ tokenizer with the ``models.vq_model.VQModel`` API.

    Layout matches the TencentARC IBQ pretrain (``encoder`` / ``decoder`` /
    ``quant_conv`` / ``post_quant_conv`` / ``quantize.embedding``) so the
    published checkpoint drops in via ``convert_ibq_pretrained_state_dict``.
    The forward returns the same outer tuple shape as VQModel so the
    Stage 0/1 training loop (``vq(..., return_distance=True)``) and the
    Stage 2 inference path consume the result without branching on type.
    """

    def __init__(self, config: IBQModelArgs):
        super().__init__()
        self.config = config

        self.encoder = Encoder(
            ch=config.ch,
            out_ch=config.out_ch,
            in_channels=config.in_channels,
            num_res_blocks=config.num_res_blocks,
            attn_resolutions=tuple(config.attn_resolutions),
            z_channels=config.z_channels,
            ch_mult=tuple(config.ch_mult),
            resolution=config.resolution,
            dropout=config.dropout_p,
            double_z=False,
        )
        self.decoder = Decoder(
            ch=config.ch,
            out_ch=config.out_ch,
            in_channels=config.in_channels,
            num_res_blocks=config.num_res_blocks,
            attn_resolutions=tuple(config.attn_resolutions),
            z_channels=config.z_channels,
            ch_mult=tuple(config.ch_mult),
            resolution=config.resolution,
            dropout=config.dropout_p,
        )
        self.quantize = IBQQuantizer(
            n_e=config.codebook_size,
            e_dim=config.codebook_embed_dim,
            beta=config.commit_loss_beta,
            use_entropy_loss=config.use_entropy_loss,
            entropy_temperature=config.entropy_loss_temperature,
            sample_minimization_weight=config.sample_minimization_weight,
            batch_maximization_weight=config.batch_maximization_weight,
            entropy_loss_ratio=config.entropy_loss_ratio,
            cosine_similarity=config.cosine_similarity,
        )
        self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

    # ---- VQModel API ------------------------------------------------------
    def encode(self, x: torch.Tensor, return_distance: bool = False):
        h = self.encoder(x)
        h = self.quant_conv(h)
        out = self.quantize(h, return_distance=return_distance)
        if return_distance:
            quant, diff, info, d = out
            return quant, diff, info, d
        quant, diff, info = out
        return quant, diff, info

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

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
# TencentARC IBQ pretrained-checkpoint converter
# =============================================================================
_IBQ_LIVE_PREFIXES = ("encoder.", "decoder.", "quant_conv.", "post_quant_conv.", "quantize.")


def convert_ibq_pretrained_state_dict(
    state_dict: "OrderedDict[str, torch.Tensor]",
    prefer_ema: bool = True,
) -> "OrderedDict[str, torch.Tensor]":
    """Translate a TencentARC IBQ Lightning ``state_dict`` into one that
    ``IBQModel.load_state_dict(..., strict=False)`` accepts.

    The IBQ ckpt contains:

      * ``encoder.*`` / ``decoder.*`` / ``quant_conv.*`` /
        ``post_quant_conv.*`` / ``quantize.embedding.*`` -- live params
      * ``model_ema.decay`` / ``model_ema.num_updates``  -- EMA bookkeeping
      * ``model_ema.<key with dots stripped>``           -- EMA shadows of the
        live params (LitEma stores names without ``.`` because they would
        otherwise be parsed as submodule attributes)
      * (optional) ``loss.*`` / ``inception*`` / ``lpips*`` -- training-time
        helpers that have no analogue in our IBQModel

    Strategy mirrors ``lfq_model.convert_magvit2_pretrained_state_dict`` but
    keeps the quantizer / quant_conv / post_quant_conv params too (IBQ has a
    learnable codebook, unlike single-codebook LFQ).

    Parameters
    ----------
    state_dict : OrderedDict[str, Tensor]
        The raw lightning state_dict (i.e. ``ckpt['state_dict']``).
    prefer_ema : bool
        Whether to prefer the EMA copy over the live params when both are
        present. The IBQ inference pipeline uses EMA by default (see
        ``ibqgan.VQModel.init_from_ckpt``'s ``stage='transformer'`` branch);
        we match that default.
    """
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    stripped_to_real: dict = {}

    for k, v in state_dict.items():
        if k.startswith("model_ema."):
            continue
        if k.startswith(_IBQ_LIVE_PREFIXES):
            out[k] = v
            stripped_to_real[k.replace(".", "")] = k

    if not prefer_ema:
        return out

    for k, v in state_dict.items():
        if not k.startswith("model_ema."):
            continue
        sub = k[len("model_ema."):]
        if sub in ("decay", "num_updates"):
            continue
        real = stripped_to_real.get(sub)
        if real is None:
            # EMA shadow without a matching live param -- silently drop.
            continue
        out[real] = v

    return out


def load_ibq_pretrained_(
    model: IBQModel,
    ckpt_path: str,
    prefer_ema: bool = True,
    map_location: str = "cpu",
):
    """Convenience helper: load a TencentARC IBQ ``.ckpt`` directly into an
    instantiated :class:`IBQModel`. Returns ``(missing, unexpected)`` from
    ``load_state_dict``.
    """
    raw = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    sd = raw["state_dict"] if "state_dict" in raw else raw
    converted = convert_ibq_pretrained_state_dict(sd, prefer_ema=prefer_ema)
    return model.load_state_dict(converted, strict=False)


# =============================================================================
# IBQ Model factory + registry (mirrors VQ_models in models/vq_model.py)
# =============================================================================
def IBQ_16(**kwargs):
    """IBQ tokenizer, downsample ratio 16, ch=128, 4 res blocks per level.

    Mirrors the TencentARC ``pretrain_ibqgan_16384.yaml`` architecture:
    ``ch=128, ch_mult=[1,1,2,2,4], num_res_blocks=4, attn_resolutions=[16],
    z_channels=256``, with a learnable codebook of size 16384 x 256. The
    ``codebook_size`` / ``codebook_embed_dim`` kwargs are forwarded so the
    Stage 0/1/2 scripts can keep their ``--codebook-size`` /
    ``--codebook-embed-dim`` flags exactly as for VQ / LFQ.
    """
    args = dict(
        codebook_size=16384,
        codebook_embed_dim=256,
        z_channels=256,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=4,
        attn_resolutions=[16],
    )
    args.update(kwargs)
    # For IBQ the codebook embed dim and the encoder z_channels are tied
    # (quant_conv is a 1x1 conv between them, but the pretrain uses
    # embed_dim == z_channels == 256). Keep them in sync if the caller only
    # overrode one of the two.
    return IBQModel(IBQModelArgs(**args))


IBQ_models = {"IBQ-16": IBQ_16}
