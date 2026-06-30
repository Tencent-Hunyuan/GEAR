"""Self-contained GPIC **FD-DINOv2** evaluator (DINOv2 only).

A single-file port of the DINO path of the GPIC eval toolkit
(``gpic/gpic_eval``, which itself builds on dgm-eval / clovaai-prdc), so the
metrics live entirely in this repo and we don't depend on / commit the external
gpic repo. The numbers match ``gpic-eval ... --models dino`` because the metric
math (FD, PRDC, MMD) and the DINOv2 preprocessing are copied verbatim.

What it computes (all on DINOv2 ViT-L/14, 1024-d features):
  * **FD**   -- Frechet Distance (FD-DINOv2; the headline GPIC metric)
  * **PRDC** -- Precision / Recall / Density / Coverage  (needs ref embeddings)
  * **MMD**  -- polynomial-kernel Maximum Mean Discrepancy (needs ref embeddings)

Reference (second arg) is a precomputed stats ``.npz`` (e.g. GPIC's
``reference_stats/test_stats.npz``) carrying ``dino_mu`` / ``dino_sigma`` (for
FD) and optionally ``dino_embeddings`` (needed for PRDC / MMD). The sample
(first arg) is what we generate: a directory of images, a ``.npy`` (NHWC uint8),
or a ``.npz`` with ``arr_0``.

Usage::

    python src/eval/gpic/gpic_eval_dino.py \\
        /path/to/gpic_infer_out/<exp>/<step>/cfg1-temp1/samples.npz \\
        /path/to/gpic/reference_stats/test_stats.npz \\
        --metrics fd,prdc,mmd

Acknowledgements: https://github.com/keshik6/gpic,
https://github.com/layer6ai-labs/dgm-eval,
https://github.com/clovaai/generative-evaluation-prdc.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DINO_HUB_NAME = "dinov2_vitl14"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ALL_METRICS = ["fd", "prdc", "mmd"]
_MAX_PRDC_SAMPLES = 10_000
_RECOMMENDED_SAMPLES = 50_000


# =============================================================================
# Sample loading (directory of images / .npy / .npz with arr_0)
# =============================================================================
def _dino_transform(arr):
    """HWC uint8 numpy -> [3, 224, 224] float tensor (resize 224 bicubic + IN norm)."""
    import torchvision.transforms as TF

    return TF.Compose([
        TF.Resize((224, 224), TF.InterpolationMode.BICUBIC),
        TF.ToTensor(),
        TF.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])(Image.fromarray(arr))


class _ImageArrayDataset(Dataset):
    """Indexes into an in-memory NHWC uint8 array loaded from .npy/.npz(arr_0)."""

    def __init__(self, path, transform):
        self.arr = np.load(path) if path.endswith(".npy") else np.load(path)["arr_0"]
        self.transform = transform

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, idx):
        return self.transform(self.arr[idx])


class _ImagePathDataset(Dataset):
    """Lazily loads images from a directory tree."""

    _EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, root_dir, transform):
        self.paths = sorted(
            os.path.join(dp, fn)
            for dp, _, fns in os.walk(root_dir)
            for fn in fns
            if os.path.splitext(fn)[1].lower() in self._EXTENSIONS
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = np.array(Image.open(self.paths[idx]).convert("RGB"), dtype=np.uint8)
        return self.transform(x)


def _make_sample_dataset(path: str) -> Dataset:
    if os.path.isdir(path):
        return _ImagePathDataset(path, _dino_transform)
    if path.endswith(".npy") or path.endswith(".npz"):
        return _ImageArrayDataset(path, _dino_transform)
    raise ValueError(
        f"unsupported sample path {path!r}: expected an image directory, "
        f".npy (NHWC uint8), or .npz with arr_0."
    )


@torch.no_grad()
def extract_dino_features(path: str, device, batch_size: int, num_workers: int) -> np.ndarray:
    """Run DINOv2 ViT-L/14 over a sample batch -> [N, 1024] float32 features."""
    print(f"[dino] loading {DINO_HUB_NAME} on {device}...")
    model = torch.hub.load("facebookresearch/dinov2", DINO_HUB_NAME)
    model.eval().to(device)

    dataset = _make_sample_dataset(path)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    feats = []
    for batch in tqdm(loader, desc="DINOv2 features"):
        x = batch.to(device, non_blocking=True)
        feats.append(model(x).cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


# =============================================================================
# Metrics (copied verbatim from gpic_eval / dgm-eval / clovaai-prdc)
# =============================================================================
def compute_FD_with_stats(mu1, mu2, sigma1, sigma2, eps=1e-6):
    """Frechet Distance between N(mu1, sigma1) and N(mu2, sigma2).

    Returns (fd, mean_term, cov_term) with fd = mean_term + cov_term.
    """
    from scipy import linalg

    assert mu1.shape == mu2.shape, "mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "covariances have different dimensions"

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        warnings.warn(
            f"fd calculation produces singular product; adding {eps} to diagonal",
            RuntimeWarning, stacklevel=2,
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    mean_term = diff.dot(diff)
    cov_term = np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return mean_term + cov_term, mean_term, cov_term


def _compute_pairwise_distance(data_x, data_y=None):
    import sklearn.metrics
    if data_y is None:
        data_y = data_x
    return sklearn.metrics.pairwise_distances(
        data_x, data_y, metric="euclidean", n_jobs=8)


def _get_kth_value(unsorted, k, axis=-1):
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    return k_smallests.max(axis=axis)


def _nn_distances(input_features, nearest_k):
    distances = _compute_pairwise_distance(input_features)
    return _get_kth_value(distances, k=nearest_k + 1, axis=-1)


def compute_prdc(real_features, fake_features, nearest_k=5):
    """Precision / Recall / Density / Coverage between two manifolds."""
    real_nn = _nn_distances(real_features, nearest_k)
    fake_nn = _nn_distances(fake_features, nearest_k)
    dist_rf = _compute_pairwise_distance(real_features, fake_features)

    precision = (dist_rf < np.expand_dims(real_nn, axis=1)).any(axis=0).mean()
    recall = (dist_rf < np.expand_dims(fake_nn, axis=0)).any(axis=1).mean()
    density = (1.0 / float(nearest_k)) * (
        dist_rf < np.expand_dims(real_nn, axis=1)).sum(axis=0).mean()
    coverage = (dist_rf.min(axis=1) < real_nn).mean()
    return dict(precision=precision, recall=recall, density=density, coverage=coverage)


def _mmd2(K_XX, K_XY, K_YY):
    m = K_XX.shape[0]
    Kt_XX_sum = (K_XX.sum(axis=1) - np.diagonal(K_XX)).sum()
    Kt_YY_sum = (K_YY.sum(axis=1) - np.diagonal(K_YY)).sum()
    K_XY_sum = K_XY.sum(axis=0).sum()
    mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
    mmd2 -= 2 * K_XY_sum / (m * m)
    return mmd2


def _polynomial_mmd(feat_r, feat_gen, degree=3, gamma=None, coef0=1):
    from sklearn.metrics.pairwise import polynomial_kernel
    K_XX = polynomial_kernel(feat_r, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(feat_gen, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(feat_r, feat_gen, degree=degree, gamma=gamma, coef0=coef0)
    return _mmd2(K_XX, K_XY, K_YY)


def compute_mmd(feat_real, feat_gen, n_subsets=100, subset_size=1000):
    subset_size = min(subset_size, feat_real.shape[0], feat_gen.shape[0])
    mmds = np.zeros(n_subsets)
    choice = np.random.choice
    with tqdm(range(n_subsets), desc="MMD") as bar:
        for i in bar:
            g = feat_real[choice(len(feat_real), subset_size, replace=False)]
            r = feat_gen[choice(len(feat_gen), subset_size, replace=False)]
            mmds[i] = _polynomial_mmd(g, r)
            bar.set_postfix({"mean": mmds[: i + 1].mean()})
    return mmds


# =============================================================================
# Orchestration
# =============================================================================
def _fd_stats_from_features(features: np.ndarray):
    feats = features.astype(np.float64)
    return np.mean(feats, axis=0), np.cov(feats, rowvar=False)


def evaluate(sample_path, ref_path, metrics, batch_size, num_workers, device):
    ref = np.load(ref_path)
    ref_keys = set(ref.keys())
    if "dino_mu" not in ref_keys or "dino_sigma" not in ref_keys:
        raise KeyError(
            f"{ref_path} lacks dino_mu/dino_sigma. Found: {sorted(ref_keys)}"
        )
    ref_mu, ref_sigma = ref["dino_mu"], ref["dino_sigma"]
    ref_emb = ref["dino_embeddings"] if "dino_embeddings" in ref_keys else None

    need_features = ("prdc" in metrics) or ("mmd" in metrics)
    if need_features and ref_emb is None:
        warnings.warn(
            "reference has no `dino_embeddings`; PRDC/MMD will be skipped "
            "(only FD is possible from mu/sigma).", stacklevel=2,
        )

    sample_feats = extract_dino_features(sample_path, device, batch_size, num_workers)
    print(f"[dino] sample features: {sample_feats.shape}")
    if ref_emb is not None:
        print(f"[dino] reference embeddings: {ref_emb.shape}")
    if len(sample_feats) < _RECOMMENDED_SAMPLES:
        warnings.warn(
            f"only {len(sample_feats)} sample features; "
            f"{_RECOMMENDED_SAMPLES} recommended for a reliable FD-DINOv2.",
            stacklevel=2,
        )

    results: dict[str, float | None] = {}

    if "fd" in metrics:
        s_mu, s_sigma = _fd_stats_from_features(sample_feats)
        fd, fd_mu, fd_sigma = compute_FD_with_stats(s_mu, ref_mu, s_sigma, ref_sigma)
        results["fd"] = float(fd)
        results["fd_mu"] = float(fd_mu)
        results["fd_sigma"] = float(fd_sigma)

    if "mmd" in metrics and ref_emb is not None:
        mmds = compute_mmd(ref_emb, sample_feats)
        results["mmd"] = float(np.mean(mmds))

    if "prdc" in metrics and ref_emb is not None:
        n = min(_MAX_PRDC_SAMPLES, len(ref_emb), len(sample_feats))
        ref_sub = ref_emb
        sample_sub = sample_feats
        if len(ref_emb) > n:
            ref_sub = ref_emb[np.random.choice(len(ref_emb), n, replace=False)]
        if len(sample_feats) > n:
            sample_sub = sample_feats[np.random.choice(len(sample_feats), n, replace=False)]
        prdc = compute_prdc(ref_sub, sample_sub, nearest_k=5)
        results["precision"] = float(prdc["precision"])
        results["recall"] = float(prdc["recall"])
        results["density"] = float(prdc["density"])
        results["coverage"] = float(prdc["coverage"])

    return results


def _print_table(results: dict):
    order = ["fd", "fd_mu", "fd_sigma", "precision", "recall", "density", "coverage", "mmd"]
    labels = {
        "fd": "FD-DINOv2", "fd_mu": "FD (mu component)", "fd_sigma": "FD (sigma component)",
        "precision": "Precision", "recall": "Recall", "density": "Density",
        "coverage": "Coverage", "mmd": "MMD",
    }
    rows = [(labels[k], f"{results[k]:.4f}") for k in order if k in results]
    if not rows:
        return
    w1 = max(len(r[0]) for r in rows) + 2
    w2 = max(len(r[1]) for r in rows) + 2
    bar = f"+{'-' * w1}+{'-' * w2}+"
    print("\n" + bar)
    print(f"|{'Metric':<{w1}}|{'Value':<{w2}}|")
    print(bar)
    for name, val in rows:
        print(f"|{name:<{w1}}|{val:<{w2}}|")
    print(bar)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("sample", help="generated images: directory, .npy (NHWC uint8), or .npz(arr_0).")
    p.add_argument("reference", help="reference stats .npz with dino_mu/dino_sigma (+ dino_embeddings).")
    p.add_argument("--metrics", type=str, default="fd,prdc,mmd",
                   help=f"comma-separated subset of {ALL_METRICS}. Default: all.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", type=str, default=None,
                   help="e.g. cuda:0. Default: auto (cuda if available).")
    p.add_argument("--out-dir", type=str, default="out_gpic-eval",
                   help="directory to write the results JSON into.")
    args = p.parse_args()

    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    bad = [m for m in metrics if m not in ALL_METRICS]
    if bad:
        raise ValueError(f"unknown metric(s): {bad}. Available: {ALL_METRICS}")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        warnings.warn("running DINOv2 on CPU (no GPU detected); pass --device cuda:0.",
                      stacklevel=2)

    results = evaluate(args.sample, args.reference, metrics,
                       args.batch_size, args.num_workers, device)
    _print_table(results)

    os.makedirs(args.out_dir, exist_ok=True)
    s_name = os.path.splitext(os.path.basename(args.sample.rstrip("/")))[0]
    r_name = os.path.splitext(os.path.basename(args.reference))[0]
    out_path = os.path.join(args.out_dir, f"{s_name}_{r_name}.json")
    with open(out_path, "w") as f:
        json.dump({"dinov2": results}, f, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
