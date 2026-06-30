"""Normalization constants shared across the REPA target encoders."""

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD  # noqa: F401

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

# SigLIP / SigLIP2 use a symmetric [-1, 1] normalisation: mean = std = 0.5
# (equivalent to ``(x - 127.5) / 127.5`` on uint8 input).
SIGLIP_DEFAULT_MEAN = (0.5, 0.5, 0.5)
SIGLIP_DEFAULT_STD = (0.5, 0.5, 0.5)

__all__ = [
    "CLIP_DEFAULT_MEAN",
    "CLIP_DEFAULT_STD",
    "IMAGENET_DEFAULT_MEAN",
    "IMAGENET_DEFAULT_STD",
    "SIGLIP_DEFAULT_MEAN",
    "SIGLIP_DEFAULT_STD",
]
