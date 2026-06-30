"""REPA target-encoder registry for ``src``.

Each per-encoder module under this package exposes three functions with the
same shape:

* ``load(model_config, device, resolution) -> nn.Module``
    Returns an encoder with a public ``embed_dim`` attribute and a
    ``forward_features(x) -> Any`` method whose output is unpacked by
    ``extract``.

* ``preprocess(x_uint8, resolution) -> Tensor``
    Takes the raw ``(B, 3, H, W) uint8`` images already produced by the
    train pipeline and returns the float tensor that should be fed to
    ``encoder.forward_features``.

* ``extract(out) -> Tensor``
    Pulls the spatial patch tokens (``(B, L, D)``) out of one encoder
    forward pass.

The top-level :func:`load_encoders` / :func:`preprocess_raw_image` /
:func:`extract_repa_target` functions in :mod:`.registry` dispatch to the
right per-encoder module based on the ``<type>-<arch>-<config>`` triple
parsed from the ``--enc-type`` CLI flag.
"""

from .layer_extraction import (
    extract_at_layer,
    num_encoder_layers,
    spatial_norm,
)
from .registry import (
    extract_repa_target,
    load_encoders,
    preprocess_raw_image,
    supports_resolution,
)

__all__ = [
    "extract_at_layer",
    "extract_repa_target",
    "load_encoders",
    "num_encoder_layers",
    "preprocess_raw_image",
    "spatial_norm",
    "supports_resolution",
]
