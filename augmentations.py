"""Albumentations pipelines for 28x28 grayscale MNIST digits (albumentations 2.0.8).

Every pipeline maps raw uint8 [28, 28] -> float32 [28, 28] in this order:

    [train only] geometric ops on uint8, fill=0 (black, i.e. MNIST background)
    Normalize((x / 255 - 0.1307) / 0.3081)   <- exactly once, always last

Modes:
    none            deterministic: Normalize only (also used for every val/test image)
    affine          conservative rotation / translation / per-axis scale / shear
    affine_elastic  affine + mild elastic distortion

Sources:
- Affine distortions of MNIST digits (translation, scaling, squeezing, shearing) go back to
  LeCun, Bottou, Bengio & Haffner, "Gradient-Based Learning Applied to Document Recognition",
  Proc. IEEE 1998.
- Elastic distortions were introduced later by Simard, Steinkraus & Platt, "Best Practices for
  Convolutional Neural Networks Applied to Visual Document Analysis", ICDAR 2003.
- Library: Buslaev et al., "Albumentations: Fast and Flexible Image Augmentations", Information 2020.

Elastic parameters are *albumentations 2.0.8* parameters, which this file does not equate with
the paper's values. In 2.0.8, with noise_distribution="uniform", the displacement field is
`alpha * GaussianBlur(uniform(-1, 1) noise, sigma)` in pixels, blurred with an automatic kernel
(approximate=False). `inspect_data.py` measures the resulting displacement in pixels.

Seeding: Compose owns its RNG, so np.random.seed does not affect it. Training augmentation is a
*pure function of (augment seed, epoch, manifest row)*. The sampler yields (row, epoch) pairs, and
DigitSumDataset reseeds the transform with derive_seed(augment_seed, epoch, row) before
transforming that row's images. As a result:
- num_workers, persistent workers, batch size, and resume do not change which augmentation a row
  gets, so runs that differ only in num_workers see identical data;
- different rows and epochs get different seeds, so this is not the "same random transform for
  every sample" bug (which comes from reseeding with one constant);
- the images within a row draw sequentially from that row's stream.
`seed_compose` gives Compose and all child transforms ONE shared generator pair. Compose's own
`set_random_seed` instead gives every child an identically seeded generator, so Affine and
ElasticTransform would replay the same random numbers.
"""

from __future__ import annotations

import random

import albumentations as A
import cv2
import numpy as np
import torch

# Classic MNIST statistics (computed over the 60k official training images). inspect_data.py
# re-measures them on our training pool.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
NORMALIZED_BLACK = (0.0 - MNIST_MEAN) / MNIST_STD  # -0.42421: value of a background pixel
NORMALIZED_WHITE = (1.0 - MNIST_MEAN) / MNIST_STD  # 2.82149


def derive_seed(*parts: int) -> int:
    """Independent 32-bit seed for a named purpose, e.g. derive_seed(seed, epoch, row)."""
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


def seed_compose(compose: A.Compose, seed: int) -> None:
    """Seed a Compose so it and all its children draw from ONE shared generator pair.

    albumentations' Compose.set_random_seed(seed) seeds every child with the same value, so two
    random children (e.g. Affine and ElasticTransform) would produce identical draws. Sharing one
    numpy Generator and one random.Random makes their draws sequential instead.
    """
    seed = int(seed) % 2**32
    compose.set_random_state(np.random.default_rng(seed), random.Random(seed))


class MnistTransform:
    """Callable wrapper: raw uint8 [28, 28] -> normalized float32 [28, 28]."""

    def __init__(self, compose: A.Compose, mode: str, seed: int | None = None):
        self.compose = compose
        self.mode = mode
        if seed is not None:
            self.set_random_seed(seed)

    def __call__(self, raw_pixels: np.ndarray) -> np.ndarray:
        if raw_pixels.dtype != np.uint8 or raw_pixels.ndim != 2:
            raise ValueError(f"Expected raw uint8 [H, W], got {raw_pixels.dtype} {raw_pixels.shape}")
        output = self.compose(image=raw_pixels)["image"]
        if output.dtype != np.float32 or output.shape != raw_pixels.shape:
            raise RuntimeError(f"Transform produced {output.dtype} {output.shape}; expected float32 {raw_pixels.shape}")
        return output

    def set_random_seed(self, seed: int) -> None:
        """Reseed Albumentations' generators (one stream shared by Compose and every child)."""
        seed_compose(self.compose, seed)


def geometric_transforms(aug_cfg, mode: str) -> list:
    """The random uint8 ops for a mode (empty for 'none'). Fill is black before normalization."""
    ops = []
    if mode in ("affine", "affine_elastic"):
        ops.append(
            A.Affine(
                rotate=(-aug_cfg.rotate_degrees, aug_cfg.rotate_degrees),
                translate_px=(-aug_cfg.translate_px, aug_cfg.translate_px),
                scale=(aug_cfg.scale_min, aug_cfg.scale_max),  # independent x/y factors
                shear=(-aug_cfg.shear_degrees, aug_cfg.shear_degrees),  # independent x/y shear
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=aug_cfg.affine_p,
            )
        )
    if mode == "affine_elastic":
        ops.append(
            A.ElasticTransform(
                alpha=aug_cfg.elastic_alpha,
                sigma=aug_cfg.elastic_sigma,
                noise_distribution="uniform",  # closest to Simard et al.'s uniform(-1, 1) fields
                approximate=False,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=aug_cfg.elastic_p,
            )
        )
    if mode not in ("none", "affine", "affine_elastic"):
        raise ValueError(f"Unknown augmentation mode {mode!r}")
    return ops


def build_transform(aug_cfg, train: bool, seed: int | None = None, mode: str | None = None) -> MnistTransform:
    """Training: configured mode. Evaluation (train=False): deterministic Normalize only."""
    chosen_mode = (mode or aug_cfg.mode) if train else "none"
    ops = geometric_transforms(aug_cfg, chosen_mode)
    ops.append(A.Normalize(mean=(MNIST_MEAN,), std=(MNIST_STD,), max_pixel_value=255.0))
    return MnistTransform(A.Compose(ops), chosen_mode, seed=seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: give each worker distinct, reproducible RNG streams.

    Training rows are reseeded per (epoch, row) by the dataset, so this matters only for accesses
    that carry no epoch (plain integer indices). Without it, every worker would start from an
    identical copy of the transform's RNG and replay the same "random" augmentations.
    torch sets each worker's seed to base_seed + worker_id.
    """
    info = torch.utils.data.get_worker_info()
    seed = int(info.seed) % 2**32
    random.seed(seed)
    np.random.seed(seed)
    transform = getattr(info.dataset, "transform", None)
    if transform is not None and hasattr(transform, "set_random_seed"):
        transform.set_random_seed(seed)


def measure_elastic_displacement(alpha: float, sigma: float, trials: int = 200, seed: int = 0, margin: int = 4) -> dict:
    """Measure the elastic displacement (pixels) produced by the installed library.

    Probe: a 3-channel float image (x coordinate, y coordinate, ones). A warp samples the source
    location, and bilinear interpolation of a linear ramp is exact, so output - input is the
    displacement field. Pixels that sampled outside the image get the fill value, so their
    "ones" channel drops below 1; they are counted separately and excluded from the statistics,
    as are pixels within `margin` of the border.
    """
    height, width = 28, 28
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    probe = np.stack([xs, ys, np.ones_like(xs)], axis=-1)  # [28, 28, 3]
    elastic = A.Compose(
        [
            A.ElasticTransform(
                alpha=alpha,
                sigma=sigma,
                noise_distribution="uniform",
                approximate=False,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=1.0,
            )
        ],
        seed=seed,
    )
    inner = (slice(margin, height - margin), slice(margin, width - margin))
    magnitudes, per_axis, outside, total = [], [], 0, 0
    for _ in range(trials):
        warped = elastic(image=probe)["image"][inner]  # [H', W', 3]
        valid = warped[..., 2] > 0.999  # sampled fully inside the source image
        displacement = (warped[..., :2] - probe[inner][..., :2])[valid]  # [n_valid, 2]
        outside += int((~valid).sum())
        total += valid.size
        per_axis.append(displacement)
        magnitudes.append(np.linalg.norm(displacement, axis=-1))
    magnitudes = np.concatenate(magnitudes)
    per_axis = np.concatenate(per_axis)
    return {
        "alpha": float(alpha),
        "sigma": float(sigma),
        "rms_per_axis_px": float(np.sqrt(np.mean(per_axis**2))),
        "mean_magnitude_px": float(magnitudes.mean()),
        "p95_magnitude_px": float(np.percentile(magnitudes, 95)),
        "max_magnitude_px": float(magnitudes.max()),
        "fraction_sampled_outside": outside / total,
    }
