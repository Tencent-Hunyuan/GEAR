"""Classical ImageFolder loader for ImageNet-style class-conditional training.

Replaces the previous REPA-E H5 layout. Rationale:

* The H5 cache was originally there for I/O speed and to avoid online
  resize. With our batch sizes (BS=256 / 64 GPUs -> 4 images/GPU/step) the
  on-disk JPEG decode is not the bottleneck.
* The H5 stored a *static* center-cropped 256x256 PNG per image, so every
  epoch saw the exact same pixels per sample. With per-step random crop
  (and optional hflip) we get the canonical "different view per epoch"
  augmentation that DiT / LlamaGen / MAR all rely on.

Output contract (kept identical to the old ``CustomINH5Dataset``):

* ``image``: ``torch.uint8`` tensor of shape ``(3, image_size, image_size)``
  in the ``[0, 255]`` range. This is what ``preprocess_imgs_for_codec``
  (-> ``[-1, 1]`` for the VQ) and ``preprocess_raw_image`` (``/255`` then
  imagenet mean/std for DINOv2 / CLIP) both expect.
* ``label``: Python ``int`` class index. The DataLoader's default collate
  turns it into ``torch.long`` for the batch -- same as before.
"""

from __future__ import annotations

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def build_imagenet_dataset(
    root: str,
    image_size: int = 256,
    random_hflip: bool = True,
) -> Dataset:
    """Build the ImageNet train ``Dataset`` used by Stage 1 / Stage 2.

    Pipeline (per ``__getitem__``):

    1. ``transforms.Resize(image_size, BICUBIC)`` -- short-side resize.
       Passing an ``int`` to torchvision's ``Resize`` scales the *smaller*
       edge to ``image_size`` while preserving aspect ratio, so
       ``RandomCrop`` always succeeds without padding.
    2. ``transforms.RandomCrop(image_size)`` -- random ``image_size`` x
       ``image_size`` window. This is what restores per-epoch sample
       diversity (the H5 path stored a single fixed center crop).
    3. ``transforms.RandomHorizontalFlip(p=0.5)`` -- the standard ImageNet
       generation augmentation. Effectively doubles the dataset.
    4. ``transforms.PILToTensor()`` -- returns ``torch.uint8`` ``(3, H, W)``
       in ``[0, 255]``. We deliberately use ``PILToTensor`` instead of
       ``ToTensor``; the latter would float-normalize to ``[0, 1]`` and
       break the downstream ``raw_image / 255.0`` calls inside
       ``preprocess_raw_image``.

    Parameters
    ----------
    root
        Path to the ImageNet train folder, e.g.
        ``/.../OpenDataLab___ImageNet-1K/raw/ImageNet-1K/train``.
        Must follow the standard ``<root>/<synset>/<file>.JPEG`` layout
        (which is what ``torchvision.datasets.ImageFolder`` consumes).
    image_size
        Output crop size in pixels. Defaults to 256 (matches the rest of
        the codebase / VQ downsample assumption).
    random_hflip
        Whether to apply 50% random horizontal flip. Defaults to True --
        match LlamaGen / DiT / MAR. Set False for ablation runs.
    """
    tlist = [
        transforms.Resize(
            image_size, interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomCrop(image_size),
    ]
    if random_hflip:
        tlist.append(transforms.RandomHorizontalFlip(p=0.5))
    tlist.append(transforms.PILToTensor())
    return ImageFolder(root, transform=transforms.Compose(tlist))


def build_imagenet_val_dataset(
    root: str,
    image_size: int = 256,
    resize_mode: str = "bicubic",
) -> Dataset:
    """Deterministic ImageNet **val** loader for VQ reconstruction metrics.

    Pipeline (per ``__getitem__``):

    1. ``transforms.Resize(image_size, <resize_mode>)`` -- short-side resize.
    2. ``transforms.CenterCrop(image_size)`` -- deterministic center crop.
       We deliberately do NOT use RandomCrop / hflip here: validation must
       see the *same* pixels every eval so PSNR / SSIM / FID curves track
       real model improvement, not augmentation noise.
    3. ``transforms.PILToTensor()`` -- ``torch.uint8 (3, H, W)`` in
       ``[0, 255]`` (same contract as the train dataset; downstream
       ``preprocess_imgs_for_codec`` works unchanged).

    Parameters
    ----------
    root
        Path to the ImageNet val folder. The default in `src/scripts`
        points at:
        ``/.../OpenDataLab___ImageNet-1K/raw/ImageNet-1K/val``.
    image_size
        Eval crop size. Should match the training ``--image-size``.
    resize_mode
        Interpolation mode for the short-side resize. Two valid values:

        * ``"bicubic"`` (default) -- matches our train-time pipeline, so
          train and eval see the same pixel distribution. Use this for
          tracking checkpoints we trained ourselves.
        * ``"bilinear"`` -- matches the SEED-Voken / Open-MAGVIT2
          reference eval (``torchvision.transforms.Resize`` default is
          BILINEAR; their data loader uses it). Use this only when
          directly reproducing the published MAGVIT2 LFQ benchmark
          numbers, since on a single 256-image head-slice the
          bicubic→bilinear swap alone moves PSNR by +0.7 dB and SSIM by
          +0.03 just by changing the GT pixels.
    """
    interp = {
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "bilinear": transforms.InterpolationMode.BILINEAR,
    }.get(resize_mode.lower())
    if interp is None:
        raise ValueError(
            f"resize_mode must be 'bicubic' or 'bilinear', got {resize_mode!r}"
        )
    tlist = [
        transforms.Resize(image_size, interpolation=interp),
        transforms.CenterCrop(image_size),
        transforms.PILToTensor(),
    ]
    return ImageFolder(root, transform=transforms.Compose(tlist))
