"""DINOv2 REPA target encoder loader.

Loaded via ``torch.hub.load('facebookresearch/dinov2', ...)``. The pretrained
encoder ships with absolute positional embeddings sized for the original
224 / 14 = 16-patch training grid, so we resample them to the runtime grid
(``resolution // 16``) with ``timm.layers.pos_embed.resample_abs_pos_embed``
to support 256x256, 384x384 and 512x512 (DINOv2 sees a ``grid*14``-px image:
224 / 336 / 448 respectively).

Mirrors the original REPA-E ``preprocess_raw_image`` / ``extract_repa_target``
recipe.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.transforms import Normalize

from ._constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def load(model_config: str, device, resolution: int = 256, *, with_registers: bool = False):
    """Load a DINOv2 encoder via ``torch.hub.load``.

    Parameters
    ----------
    model_config : str
        DINOv2 size letter; one of ``"s"`` (vits14), ``"b"`` (vitb14),
        ``"l"`` (vitl14), ``"g"`` (vitg14).
    device : torch.device | str
        Where to place the encoder.
    resolution : int
        Input image resolution. The model's ``pos_embed`` is resampled to a
        ``resolution / 16``-wide grid (e.g. 16x16 @ 256, 32x32 @ 512).
    with_registers : bool
        Use the ``_reg`` checkpoints (4 extra register tokens).
    """
    import timm  # local import keeps module-level deps thin

    hub_name = f"dinov2_vit{model_config}14_reg" if with_registers else f"dinov2_vit{model_config}14"
    # ``skip_validation=True`` bypasses the ``api.github.com`` fork-check
    # which is rate-limited to 60 req/hr per IP for anonymous callers.
    encoder = torch.hub.load(
        "facebookresearch/dinov2", hub_name,
        trust_repo=True, skip_validation=True,
    )
    del encoder.head
    # DINOv2 is ViT-B/14. REPA aligns its patch tokens 1:1 with the VQ latent
    # grid, which is `resolution // 16` (the VQ downsample). We therefore size
    # the pos_embed grid to that latent grid and feed the encoder a
    # (grid * 14)-px image in `preprocess` (e.g. 256->16->224, 384->24->336,
    # 512->32->448). NB: this assumes a VQ downsample ratio of 16.
    patch_resolution = int(resolution) // 16
    encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
        encoder.pos_embed.data, [patch_resolution, patch_resolution],
    )
    encoder.head = nn.Identity()
    return encoder.to(device).eval()


def preprocess(x: torch.Tensor, resolution: int) -> torch.Tensor:
    """uint8 [0, 255] -> ImageNet-normalized float, then bicubic to (grid*14)px.

    ``grid = resolution // 16`` is the VQ latent grid, so the DINOv2 patch grid
    (input // 14) matches it exactly (256->224, 384->336, 512->448).
    """
    x = x / 255.0
    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    target = (int(resolution) // 16) * 14
    x = torch.nn.functional.interpolate(x, target, mode="bicubic")
    return x


def extract(out) -> torch.Tensor:
    """DINOv2's ``forward_features`` returns a dict; we want the patch tokens."""
    return out["x_norm_patchtokens"]
