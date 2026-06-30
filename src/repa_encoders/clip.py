"""CLIP REPA target encoder loader.

Wraps OpenAI CLIP's vision tower (``clip.load(...).visual``) so that
``forward(x) -> (B, L, D)`` returns patch tokens (CLS dropped). The wrapper
logic is vendored from REPA-E ``models/clip_vit.py`` so we do not depend on
a top-level ``models.clip_vit`` module being on ``sys.path``.

Because CLIP uses a learned positional embedding sized for the original
224x224 input, the input is bicubic-resized to ``224 * (resolution // 256)``;
resolutions other than 256 will silently misalign the position embedding and
are therefore *not* supported (the :mod:`.registry` guard rejects them).
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.transforms import Normalize

from ._constants import CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD


class _CLIPPatchTokenWrapper(nn.Module):
    """Return spatial patch tokens from an OpenAI CLIP ViT visual tower (no CLS)."""

    def __init__(self, visual: nn.Module):
        super().__init__()
        self.model = visual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.model.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device,
        )
        x = torch.cat([cls, x], dim=1)
        x = x + self.model.positional_embedding.to(x.dtype)
        x = self.model.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.model.transformer(x)
        return x.permute(1, 0, 2)[:, 1:]


def load(model_config: str, device, resolution: int = 256):
    """Load a CLIP ViT encoder via the ``clip`` package.

    Parameters
    ----------
    model_config : str
        CLIP ViT size letter; e.g. ``"L"`` (ViT-L/14).
    """
    import clip

    visual = clip.load(f"ViT-{model_config}/14", device="cpu")[0].visual
    encoder = _CLIPPatchTokenWrapper(visual).to(device)
    encoder.embed_dim = encoder.model.transformer.width
    encoder.forward_features = encoder.forward
    return encoder.eval()


def preprocess(x: torch.Tensor, resolution: int) -> torch.Tensor:
    """uint8 [0, 255] -> CLIP-normalized float, then bicubic to 224*r/256."""
    x = x / 255.0
    x = torch.nn.functional.interpolate(x, 224 * (int(resolution) // 256), mode="bicubic")
    x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    return x


def extract(out) -> torch.Tensor:
    """Wrapper forward already returns ``(B, L, D)`` patch tokens."""
    return out
