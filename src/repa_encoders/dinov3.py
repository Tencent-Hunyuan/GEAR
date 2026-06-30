"""DINOv3 REPA target encoder loader.

Loaded via ``torch.hub.load('facebookresearch/dinov3', 'dinov3_vit<size>16',
pretrained=True)``. The upstream loader auto-downloads the LVD-1689M
pretrained weights from ``https://dl.fbaipublicfiles.com/dinov3/...`` on
first use and reuses the torch.hub cache on later runs.

The DINOv3 ViT backbones use RoPE for positional encoding so they natively
accept *any* input resolution (no ``pos_embed`` resampling needed) -- the
:mod:`.registry` therefore allows 512px inference just like for DINOv2.

The upstream ``hubconf.py`` re-exports the segmentor / depther / detector /
classifier heads alongside the bare backbones, and those bring in extra
runtime deps (``termcolor``, etc.). We do not need any of them, but
``torch.hub.load`` imports the whole ``hubconf`` regardless, so the upstream
deps still have to be installable. As of 2025-09 this means at least
``termcolor`` (``pip install termcolor``) on top of stock ``torch`` /
``numpy``.

Both ``embed_dim`` and ``forward_features`` exposed by the upstream
``DinoVisionTransformer`` match the API ``train_gear`` expects -- the
returned dict has the same ``x_norm_patchtokens`` key as DINOv2 so
:func:`extract` is identical in spirit.
"""

from __future__ import annotations

import torch
from torchvision.transforms import Normalize

from ._constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

_HUB_NAME_MAP = {
    "s": "dinov3_vits16",
    "splus": "dinov3_vits16plus",
    "b": "dinov3_vitb16",
    "l": "dinov3_vitl16",
    "lplus": "dinov3_vitl16plus",
    "hplus": "dinov3_vith16plus",
    "7b": "dinov3_vit7b16",
}


def _hub_name(model_config: str) -> str:
    try:
        return _HUB_NAME_MAP[model_config]
    except KeyError as exc:
        raise ValueError(
            f"Unknown DINOv3 size '{model_config}'. "
            f"Expected one of {sorted(_HUB_NAME_MAP)}."
        ) from exc


def load(model_config: str, device, resolution: int = 256):
    """Load a DINOv3 ViT via ``torch.hub.load('facebookresearch/dinov3', ...)``.

    Same pattern as DINOv2: ``pretrained=True`` only. The upstream hub entry
    defaults to ``Weights.LVD1689M`` and builds the CDN URL internally. Do
    **not** pass ``weights="LVD1689M"`` as a plain string -- upstream treats
    non-enum strings as local checkpoint paths.
    """
    # NB: a fresh ``torch.hub.load(..., trust_repo=True)`` will clone
    # ``facebookresearch/dinov3@main`` into ``~/.cache/torch/hub/`` and import
    # its ``hubconf.py``. The clone is then reused for subsequent calls.
    # ``skip_validation=True`` bypasses the ``api.github.com`` fork-check that
    # is rate-limited to 60 req/hr per IP for anonymous callers (the
    # repository is well-known and trusted).
    encoder = torch.hub.load(
        "facebookresearch/dinov3",
        model=_hub_name(model_config),
        trust_repo=True,
        skip_validation=True,
    )
    return encoder.to(device).eval()


def preprocess(x: torch.Tensor, resolution: int) -> torch.Tensor:
    """uint8 [0, 255] -> ImageNet-normalized float. No resize (RoPE handles size)."""
    x = x / 255.0
    return Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)


def extract(out) -> torch.Tensor:
    """Same dict-output API as DINOv2."""
    return out["x_norm_patchtokens"]
