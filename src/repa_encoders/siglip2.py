"""SigLIP 2 REPA target encoder loader.

Loaded via ``transformers.SiglipVisionModel.from_pretrained(...)``. The
upstream Hugging Face repo ships per-resolution weights -- separate
checkpoints for the 256px and 512px variants -- so the right HF model name
is selected at :func:`load` time based on the requested ``resolution``.

Available sizes (``model_config``):

* ``b``      -> ``google/siglip2-base-patch16-{256,512}``
* ``l``      -> ``google/siglip2-large-patch16-{256,512}``
* ``so400m`` -> ``google/siglip2-so400m-patch16-{256,512}``

Patch size is 16, so a 256px crop yields 16*16 = 256 patch tokens and a
512px crop yields 32*32 = 1024 patch tokens. ``SiglipVisionModel`` does
not produce a CLS token, so ``last_hidden_state`` *is* the patch-token
stream and :func:`extract` is a no-op.

Note on normalisation: SigLIP / SigLIP2 use a symmetric ``mean = std = 0.5``
([-1, 1] range), **not** the ImageNet stats. The iREPA reference snippet
that uses ImageNet stats is technically wrong; we follow the official HF
``SiglipImageProcessor`` defaults instead.

Local weight mirror
-------------------
If the ``SIGLIP2_LOCAL_DIR`` environment variable is set, :func:`load`
looks for the weights at::

    $SIGLIP2_LOCAL_DIR/siglip2-{size}-patch16-{resolution}/

(the layout produced by ``huggingface-cli download --local-dir``) and
passes that directory to ``from_pretrained`` so no HF Hub download is
triggered. If the variable is unset we fall back to the canonical HF
model id and the standard ``~/.cache/huggingface`` cache. When the
variable *is* set but the resolved directory is missing we raise
loudly rather than silently re-downloading, so a mis-pointed mirror
does not turn into a sneaky network fetch.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torchvision.transforms import Normalize

from ._constants import SIGLIP_DEFAULT_MEAN, SIGLIP_DEFAULT_STD

_LOCAL_DIR_ENV = "SIGLIP2_LOCAL_DIR"

_SIZE_NAME_MAP = {
    "b": "base",
    "l": "large",
    "so400m": "so400m",
}


def _hf_model_id(model_config: str, resolution: int) -> str:
    try:
        size = _SIZE_NAME_MAP[model_config]
    except KeyError as exc:
        raise ValueError(
            f"Unknown SigLIP 2 size '{model_config}'. "
            f"Expected one of {sorted(_SIZE_NAME_MAP)}."
        ) from exc
    if resolution not in (256, 512):
        raise ValueError(
            f"SigLIP 2 only ships patch16 weights at 256/512px, got resolution={resolution}."
        )
    return f"google/siglip2-{size}-patch16-{resolution}"


class _SigLIP2PatchTokenWrapper(nn.Module):
    """Thin adapter so the train pipeline can call ``forward_features``.

    ``SiglipVisionModel`` returns a ``BaseModelOutputWithPooling`` whose
    ``last_hidden_state`` already contains the spatial patch tokens
    ``(B, L, D)`` -- SigLIP-style models drop the CLS token entirely.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.embed_dim = int(model.config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x).last_hidden_state

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


def _resolve_source(model_id: str) -> str:
    """Resolve a HF model id to either a local mirror dir or the id itself.

    Returns the local directory when ``SIGLIP2_LOCAL_DIR`` is set *and* the
    expected subdirectory exists; raises if the env var is set but the
    directory is missing (to surface mis-configured mirrors loudly).
    """
    root = os.environ.get(_LOCAL_DIR_ENV)
    if not root:
        return model_id
    basename = model_id.split("/", 1)[-1]  # e.g. "siglip2-base-patch16-256"
    local = Path(root) / basename
    config_path = local / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{_LOCAL_DIR_ENV}={root!r} is set but no local SigLIP 2 mirror "
            f"was found at {local!s} (missing config.json). Either unset "
            f"the env var to fall back to the HF Hub, point it at a "
            f"directory containing the per-checkpoint subdirs, or run "
            f"`huggingface-cli download --local-dir {local!s} {model_id}` "
            f"to populate it."
        )
    return str(local)


def load(model_config: str, device, resolution: int = 256):
    """Load a SigLIP 2 vision tower via ``transformers.SiglipVisionModel``.

    Weight source order:

    1. ``$SIGLIP2_LOCAL_DIR/siglip2-<size>-patch16-<res>`` if the env var
       is set (and the dir exists -- otherwise we raise).
    2. Otherwise the canonical HF Hub id ``google/siglip2-...`` (cached
       under ``~/.cache/huggingface``).
    """
    from transformers import SiglipVisionModel  # lazy: avoid top-level dep

    model_id = _hf_model_id(model_config, resolution)
    source = _resolve_source(model_id)
    model = SiglipVisionModel.from_pretrained(source)
    return _SigLIP2PatchTokenWrapper(model).to(device).eval()


def preprocess(x: torch.Tensor, resolution: int) -> torch.Tensor:
    """uint8 [0, 255] -> SigLIP-normalized float in [-1, 1].

    The position embedding is hard-sized to the model's native grid
    (256 or 512), and the train pipeline already feeds at that resolution,
    so we do not interpolate here.
    """
    x = x / 255.0
    return Normalize(SIGLIP_DEFAULT_MEAN, SIGLIP_DEFAULT_STD)(x)


def extract(out: torch.Tensor) -> torch.Tensor:
    """Wrapper forward already returns ``(B, L, D)`` patch tokens."""
    return out
