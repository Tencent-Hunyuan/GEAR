"""Top-level dispatcher for REPA target encoders.

The ``--enc-type`` CLI flag in :mod:`train_gear` / :mod:`train_ar` is a
comma-separated list of ``<type>-<arch>-<config>`` triples (e.g.
``"dinov2-vit-b"`` or ``"dinov2-vit-b,clip-vit-B"``). Each triple maps to one
of the per-encoder modules in this package via :data:`_REGISTRY`.

The three public functions -- :func:`load_encoders`,
:func:`preprocess_raw_image`, :func:`extract_repa_target` -- preserve the
original ``src/utils.py`` signatures so the trainers do not need to
change beyond importing them from the new location.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from torch import nn

from . import clip as _clip_mod
from . import dinov2 as _dinov2_mod
from . import dinov3 as _dinov3_mod
from . import siglip2 as _siglip2_mod
from . import vjepa21 as _vjepa21_mod


@dataclass(frozen=True)
class _EncoderHooks:
    load: Callable[..., nn.Module]
    preprocess: Callable[[torch.Tensor, int], torch.Tensor]
    extract: Callable[[object], torch.Tensor]
    supports_512: bool  # True iff this encoder family handles >256px without weight surgery


# Keys are the ``encoder_type`` token (first segment of each ``--enc-type``
# triple). To add a new encoder family, drop a module next to this file
# exposing ``load`` / ``preprocess`` / ``extract`` and register it here.
_REGISTRY: dict[str, _EncoderHooks] = {
    # NOTE: ``dinov2_reg`` is handled by sharing the same hooks as ``dinov2``;
    # ``with_registers=True`` is selected based on the encoder_type string in
    # :func:`load_encoders` to keep the registry flat.
    "dinov2": _EncoderHooks(
        load=_dinov2_mod.load,
        preprocess=_dinov2_mod.preprocess,
        extract=_dinov2_mod.extract,
        supports_512=True,
    ),
    "dinov2_reg": _EncoderHooks(
        load=_dinov2_mod.load,
        preprocess=_dinov2_mod.preprocess,
        extract=_dinov2_mod.extract,
        supports_512=True,
    ),
    "dinov3": _EncoderHooks(
        load=_dinov3_mod.load,
        preprocess=_dinov3_mod.preprocess,
        extract=_dinov3_mod.extract,
        supports_512=True,  # RoPE -> arbitrary input resolution
    ),
    "jepa21": _EncoderHooks(
        load=_vjepa21_mod.load,
        preprocess=_vjepa21_mod.preprocess,
        extract=_vjepa21_mod.extract,
        supports_512=True,  # RoPE + interpolate_rope -> arbitrary input resolution
    ),
    "clip": _EncoderHooks(
        load=_clip_mod.load,
        preprocess=_clip_mod.preprocess,
        extract=_clip_mod.extract,
        # CLIP ViT-L/14's learned pos_embed is sized for the 224 grid; the
        # preprocess path resizes to 224 * (resolution // 256) so only 256px
        # cleanly maps onto the original 16-patch grid.
        supports_512=False,
    ),
    "siglip2": _EncoderHooks(
        load=_siglip2_mod.load,
        preprocess=_siglip2_mod.preprocess,
        extract=_siglip2_mod.extract,
        # SigLIP 2 ships per-resolution weights (patch16-256 / patch16-512).
        # ``load`` swaps between them based on the requested ``resolution``;
        # both are supported at native quality.
        supports_512=True,
    ),
}


def _lookup(encoder_type: str) -> _EncoderHooks:
    if encoder_type not in _REGISTRY:
        raise NotImplementedError(
            f"Encoder type '{encoder_type}' is not registered. "
            f"Known encoders: {sorted(_REGISTRY)}. "
            f"To add a new one, drop a module under src/repa_encoders/ "
            f"exposing load/preprocess/extract and register it in _REGISTRY."
        )
    return _REGISTRY[encoder_type]


def supports_resolution(encoder_type: str, resolution: int) -> bool:
    """Whether ``encoder_type`` can be evaluated at ``resolution`` without surgery."""
    if resolution == 256:
        return True
    # 384 needs a non-power-of-two patch grid (24x24). Only DINOv2's
    # ``preprocess`` is wired to derive the grid as ``resolution // 16`` and
    # resample pos_embed accordingly; the other families still assume 256/512.
    if resolution == 384:
        return encoder_type in ("dinov2", "dinov2_reg")
    if resolution >= 512:
        return _lookup(encoder_type).supports_512
    return False


@torch.no_grad()
def load_encoders(enc_type: str, device, resolution: int = 256) -> Tuple[list, list, list]:
    """Load one or more REPA target encoders.

    ``enc_type`` is a comma-separated string of ``<type>-<arch>-<config>``
    triples (the original REPA-E convention). Returns
    ``(encoders, encoder_types, architectures)`` for backwards compatibility
    with :mod:`train_gear` / :mod:`train_ar`.
    """
    assert resolution in (256, 384, 512, 1024), f"Unsupported resolution {resolution}"

    encoders, architectures, encoder_types = [], [], []
    for enc_name in enc_type.split(","):
        encoder_type, architecture, model_config = enc_name.split("-")

        if not supports_resolution(encoder_type, resolution):
            raise NotImplementedError(
                f"Encoder '{encoder_type}' does not support {resolution}x{resolution}. "
                f"Only encoders with RoPE / resamplable pos_embed / per-resolution "
                f"weights are wired up for 512px "
                f"(currently: dinov2, dinov3, jepa21, siglip2)."
            )

        architectures.append(architecture)
        encoder_types.append(encoder_type)

        hooks = _lookup(encoder_type)
        if encoder_type == "dinov2_reg":
            encoder = hooks.load(model_config, device, resolution, with_registers=True)
        else:
            encoder = hooks.load(model_config, device, resolution)
        encoders.append(encoder)

    return encoders, encoder_types, architectures


def preprocess_raw_image(x: torch.Tensor, enc_type: str) -> torch.Tensor:
    """Per-encoder uint8 -> normalized-float pipeline.

    Note: ``enc_type`` here is the bare encoder family token (e.g.
    ``"dinov2"``) -- this is what :mod:`train_gear` extracts as
    ``encoder_types[i]`` from :func:`load_encoders`. The substring match is
    intentional so legacy names like ``"dinov2_reg"`` keep working.
    """
    resolution = x.shape[-1]
    # Substring matching preserves the original behaviour for variants like
    # ``dinov2_reg`` (treated as DINOv2 for preprocessing).
    for key, hooks in _REGISTRY.items():
        if key in enc_type:
            return hooks.preprocess(x, resolution)
    raise NotImplementedError(
        f"preprocess_raw_image: no recipe for '{enc_type}'. "
        f"Known encoders: {sorted(_REGISTRY)}."
    )


def extract_repa_target(zs_raw, encoder_type: str) -> torch.Tensor:
    """Pull the patch tokens out of a single encoder forward pass."""
    for key, hooks in _REGISTRY.items():
        if key in encoder_type:
            return hooks.extract(zs_raw)
    # Legacy fall-throughs from the original ``src/utils.py``.
    if "mocov3" in encoder_type:
        return zs_raw[:, 1:]
    return zs_raw
