"""V-JEPA 2.1 REPA target encoder loader.

Architecture comes from ``torch.hub.load('facebookresearch/vjepa2', ...,
pretrained=False)`` (no dependency on the local ``vjepa2/`` clone). The
pretrained weights are fetched separately because the upstream
``vjepa2/src/hub/backbones.py`` ships with ``VJEPA_BASE_URL =
"http://localhost:8300"`` (a test stub) -- ``pretrained=True`` would hit
localhost and crash. We therefore:

1. Build the encoder + (unused) predictor via the hub factory at
   ``pretrained=False``.
2. Download the official ``.pt`` from ``https://dl.fbaipublicfiles.com/vjepa2``
   with ``torch.hub.load_state_dict_from_url`` (first use downloads; later
   runs reuse ``~/.cache/torch/hub/checkpoints/``).
3. Strip the ``module.backbone.`` key prefix and ``strict``-load the EMA
   encoder weights.

The encoder forward expects ``[B, 3, T, H, W]`` and routes through its
dedicated *image* branch (``patch_embed_img`` with ``tubelet=1`` plus
``img_mod_embed``) when ``T == img_temporal_dim_size == 1``. We wrap it in
:class:`VJepa21EncoderForImages` so the train pipeline can keep feeding
``[B, 3, H, W]`` and get ``(B, L, D)`` patch tokens back.

Both the patch embed (Conv3D, tubelet=1) and RoPE position encoding accept
any input resolution, so 512px inference is supported without weight
surgery.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.transforms import Normalize

from ._constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

_GITHUB_REPO = "facebookresearch/vjepa2"
_VJEPA_CDN = "https://dl.fbaipublicfiles.com/vjepa2"

# (hub_function_name, official .pt filename without extension).
_HUB_NAME_MAP = {
    "b": ("vjepa2_1_vit_base_384", "vjepa2_1_vitb_dist_vitG_384"),
    "l": ("vjepa2_1_vit_large_384", "vjepa2_1_vitl_dist_vitG_384"),
    "g": ("vjepa2_1_vit_giant_384", "vjepa2_1_vitg_384"),
    "G": ("vjepa2_1_vit_gigantic_384", "vjepa2_1_vitG_384"),
}


def _hub_entry(model_config: str) -> tuple[str, str]:
    try:
        return _HUB_NAME_MAP[model_config]
    except KeyError as exc:
        raise ValueError(
            f"Unknown V-JEPA 2.1 size '{model_config}'. "
            f"Expected one of {sorted(_HUB_NAME_MAP)}."
        ) from exc


class VJepa21EncoderForImages(nn.Module):
    """Adapter that lets a V-JEPA 2.1 encoder consume ``[B, 3, H, W]`` images.

    The wrapped encoder is the full ``app.vjepa_2_1.models.vision_transformer
    .VisionTransformer`` built by the upstream hub factory: it owns both the
    video ``patch_embed`` (tubelet=2) and the image-only ``patch_embed_img``
    (tubelet=1), plus the ``img_mod_embed`` / ``video_mod_embed`` modality
    tokens. Inserting a ``T=1`` axis and letting the encoder's own
    ``check_temporal_dim`` route to ``patch_embed_img`` is exactly what
    matches the pretraining input distribution for single-image inference.
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = int(encoder.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) ImageNet-normalized -> (B, L, D)
        return self.encoder(x.unsqueeze(2))

    # ``train_gear`` calls ``enc.forward_features(...)``; alias to forward.
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


def _clean_state_dict(sd: dict) -> dict:
    """Strip the ``module.backbone.`` DDP prefix from V-JEPA 2.1 state_dicts."""
    out = {}
    for k, v in sd.items():
        k = k.replace("module.", "").replace("backbone.", "")
        out[k] = v
    return out


def load(model_config: str, device, resolution: int = 256):
    """Load a V-JEPA 2.1 encoder via ``torch.hub.load(facebookresearch/vjepa2, ...)``.

    Build the architecture at ``pretrained=False`` (sidesteps the broken
    upstream ``VJEPA_BASE_URL = "http://localhost:8300"`` stub) and load the
    EMA encoder weights from the official CDN via ``load_state_dict_from_url``.
    """
    hub_name, default_file = _hub_entry(model_config)

    # Step 1: architecture. ``num_frames>=2`` keeps ``is_video=True`` so the
    # video-branch ``patch_embed`` (Conv3D, tubelet=2) is constructed -- we
    # need that so the strict state_dict load covers every key.
    # ``skip_validation=True`` bypasses the ``api.github.com`` fork-check
    # (anonymous: 60 req/hr); the upstream repo is well-known and trusted.
    encoder, _predictor = torch.hub.load(
        repo_or_dir=_GITHUB_REPO,
        model=hub_name,
        pretrained=False,
        num_frames=2,
        trust_repo=True,
        skip_validation=True,
    )

    # Step 2: weights from the official CDN (torch.hub URL cache).
    weights_url = f"{_VJEPA_CDN}/{default_file}.pt"
    state_dict = torch.hub.load_state_dict_from_url(weights_url, map_location="cpu")

    # ``ema_encoder`` is the EMA shadow used by all V-JEPA 2.x downstream eval
    # configs (e.g. ``configs/eval_2_1/vitb-384/*.yaml`` set
    # ``checkpoint_key: ema_encoder``). It's preferred over the live
    # ``encoder`` because it's smoother / less noisy.
    if "ema_encoder" not in state_dict:
        raise KeyError(
            f"V-JEPA 2.1 checkpoint at {weights_url!r} has no 'ema_encoder' key "
            f"(top-level keys: {list(state_dict)[:10]})."
        )
    encoder.load_state_dict(_clean_state_dict(state_dict["ema_encoder"]), strict=True)

    return VJepa21EncoderForImages(encoder).to(device).eval()


def preprocess(x: torch.Tensor, resolution: int) -> torch.Tensor:
    """uint8 [0, 255] -> ImageNet-normalized float. No resize (RoPE handles size)."""
    x = x / 255.0
    return Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)


def extract(out: torch.Tensor) -> torch.Tensor:
    """V-JEPA 2.1 encoder already returns ``(B, L, D)`` patch tokens (no CLS)."""
    return out
