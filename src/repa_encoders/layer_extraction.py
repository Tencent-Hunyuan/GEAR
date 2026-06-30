"""Per-layer extraction + iREPA-style spatial-normalization helpers.

These let :mod:`src.train_gear` / :mod:`src.train_ar` swap
the REPA target patch tokens between

    * the default "last transformer block + final LayerNorm" output
      (what the encoders' ``forward_features`` returns), and

    * an intermediate block's *pre-norm* output, selected by 1-based
      ``layer`` index.

and optionally apply iREPA's per-channel spatial normalization on the
target before the cosine alignment (see ``iREPA/ldm/utils.py``).

These are pure helpers -- no new entries in :data:`.registry._REGISTRY`,
and they keep the existing ``encoder.forward_features`` /
``extract_repa_target`` contract intact when ``layer == -1``
(i.e. the default code path is unchanged).
"""

from __future__ import annotations

import torch
from torch import nn

from .registry import extract_repa_target


# ---------------------------------------------------------------------------
# Encoder-family introspection
# ---------------------------------------------------------------------------
def _block_list(encoder: nn.Module, encoder_type: str) -> list[nn.Module]:
    """Return the ordered list of transformer blocks (input-adjacent first).

    Mirrors the path-resolution that :mod:`tools.diagnose_repa_targets_per_layer`
    uses for diagnostics; keeping the two in sync ensures the "layer N" we
    train against is the exact same tensor we measured offline.
    """
    if "dinov2" in encoder_type or "dinov3" in encoder_type:
        return list(encoder.blocks)
    if "siglip2" in encoder_type:
        # ``SiglipVisionModel`` -> ``model.vision_model.encoder.layers``
        return list(encoder.model.vision_model.encoder.layers)
    if "jepa21" in encoder_type:
        # Our wrapper exposes the underlying ViT as ``.encoder``.
        return list(encoder.encoder.blocks)
    raise NotImplementedError(
        f"_block_list: no rule for encoder_type={encoder_type!r}. "
        f"Supported families: dinov2, dinov2_reg, dinov3, siglip2, jepa21."
    )


def _patch_slice(encoder_type: str) -> slice:
    """Slice over the token axis that keeps only spatial patch tokens.

    A transformer-block output for the registered encoders is shaped
    ``(B, n_special + L_patches, D)`` for DINO-family backbones (CLS +
    optional register/storage tokens), and ``(B, L_patches, D)`` for the
    others (no CLS).
    """
    if "dinov2_reg" in encoder_type:
        return slice(1 + 4, None)              # 1 CLS + 4 register tokens
    if "dinov2" in encoder_type:
        return slice(1, None)                  # 1 CLS, no registers
    if "dinov3" in encoder_type:
        return slice(1 + 4, None)              # 1 CLS + 4 storage tokens
    if "siglip2" in encoder_type or "jepa21" in encoder_type:
        return slice(0, None)                  # no special tokens
    raise NotImplementedError(
        f"_patch_slice: no rule for encoder_type={encoder_type!r}."
    )


def num_encoder_layers(encoder: nn.Module, encoder_type: str) -> int:
    """Number of transformer blocks for the given encoder."""
    return len(_block_list(encoder, encoder_type))


# ---------------------------------------------------------------------------
# Layer-aware target extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_at_layer(
    encoder: nn.Module,
    encoder_type: str,
    x: torch.Tensor,
    *,
    layer: int = -1,
) -> torch.Tensor:
    """Extract REPA target patch tokens from ``encoder`` at the given depth.

    Parameters
    ----------
    encoder, encoder_type, x
        The encoder, its family token (e.g. ``"dinov3"``), and the already-
        ``preprocess_raw_image``-ed input tensor.
    layer
        ``-1`` (default) -> the standard pipeline
        ``encoder.forward_features(x) -> extract_repa_target(out, encoder_type)``
        which is the **post final LayerNorm** output that REPA has always
        aligned to.

        ``>= 1`` -> hook the (``layer-1``)-th transformer block's output and
        return its patch tokens directly (**pre any final norm**, special
        tokens stripped per family). Numbering is 1-based to match
        ``tools/diagnose_repa_targets_per_layer.py``.

    Returns
    -------
    Tensor ``(B, L, D)`` of patch tokens.
    """
    if layer == -1:
        z_raw = encoder.forward_features(x)
        return extract_repa_target(z_raw, encoder_type)

    blocks = _block_list(encoder, encoder_type)
    if not (1 <= layer <= len(blocks)):
        raise ValueError(
            f"extract_at_layer: layer={layer} out of range; "
            f"{encoder_type} has {len(blocks)} blocks "
            f"(valid: -1 for post-norm, or 1..{len(blocks)} for a block index)."
        )

    captured: list[torch.Tensor] = []

    def _hook(_module, _input, output):
        # DINOv3 routes a single batch through its ``_forward_list`` path
        # and returns a length-1 ``list[Tensor]``; HF blocks may return a
        # tuple ``(hidden_states, attn_weights, ...)``. Unwrap to a Tensor.
        if isinstance(output, (list, tuple)) and len(output) > 0:
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"extract_at_layer: block-{layer} hook captured "
                f"non-tensor of type {type(output)!r}."
            )
        captured.append(output)

    handle = blocks[layer - 1].register_forward_hook(_hook)
    try:
        # NB: we still run the full forward (subsequent blocks + final norm
        # are wasted work). For ViT-B/L this is well below 5% of one training
        # step, so we trade a tiny inefficiency for universal compatibility
        # across the registered encoder families.
        _ = encoder.forward_features(x)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(
            f"extract_at_layer: forward_features for '{encoder_type}' did not "
            f"invoke block {layer} (hook never fired)."
        )
    return captured[-1][:, _patch_slice(encoder_type), :].contiguous()


# ---------------------------------------------------------------------------
# iREPA-style spatial normalization
# ---------------------------------------------------------------------------
def spatial_norm(
    z: torch.Tensor,
    *,
    mode: str = "none",
    alpha: float = 0.6,
    eps: float = 1e-6,
) -> torch.Tensor:
    """iREPA-style per-channel spatial normalization on ``(B, L, D)`` features.

    Parameters
    ----------
    z
        Patch-token tensor ``(B, L, D)``. ``L`` is the spatial axis.
    mode
        ``'none'``    : returns ``z`` unchanged.

        ``'demean'``  : ``z -= alpha * mean_l z`` (kill the per-channel
                        spatial DC component only; preserves channel scale).

        ``'zscore'``  : ``(z - alpha * mean_l z) / (std_l z + eps)``.
                        Matches the function in ``iREPA/ldm/utils.py``
                        (``SpatialNormalization`` at ``method="zscore"``)
                        and also rebalances per-channel variance.
    alpha
        Mean-subtraction strength. iREPA LDM defaults to ``0.6``, JiT to
        ``0.8``; the mathematical "kill DC entirely" value is ``1.0``.

    Returns
    -------
    Tensor with the same shape as ``z``.
    """
    if mode == "none":
        return z
    mean = z.mean(dim=1, keepdim=True)
    z = z - alpha * mean
    if mode == "demean":
        return z
    if mode == "zscore":
        std = z.std(dim=1, keepdim=True)
        return z / (std + eps)
    raise ValueError(
        f"spatial_norm: mode={mode!r} not supported; "
        f"expected 'none', 'demean', or 'zscore'."
    )


__all__ = [
    "extract_at_layer",
    "num_encoder_layers",
    "spatial_norm",
]
