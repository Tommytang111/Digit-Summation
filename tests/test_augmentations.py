import numpy as np
import pytest
import torch

from augmentations import NORMALIZED_BLACK, NORMALIZED_WHITE, build_transform, measure_elastic_displacement
from config import AugmentConfig
from dataset import DigitSumDataset, SequenceManifest

RNG = np.random.default_rng(0)
DIGIT = RNG.integers(0, 256, size=(28, 28), dtype=np.uint8)


def test_eval_transform_is_deterministic_and_normalizes_once():
    transform = build_transform(AugmentConfig(mode="affine_elastic"), train=False)  # eval ignores the mode
    assert transform.mode == "none"
    out = transform(DIGIT)
    assert out.dtype == np.float32 and out.shape == (28, 28)
    assert np.array_equal(out, transform(DIGIT))
    # Exactly one normalization: (x / 255 - mean) / std. Twice would give different constants.
    assert np.allclose(transform(np.zeros((28, 28), np.uint8)), NORMALIZED_BLACK, atol=1e-5)
    assert np.allclose(transform(np.full((28, 28), 255, np.uint8)), NORMALIZED_WHITE, atol=1e-5)


def test_rejects_non_uint8_input():
    with pytest.raises(ValueError):
        build_transform(AugmentConfig(), train=False)(DIGIT.astype(np.float32))


@pytest.mark.parametrize("mode", ["affine", "affine_elastic"])
def test_train_transforms_shape_range_and_seeding(mode):
    cfg = AugmentConfig(mode=mode, elastic_p=1.0)
    first = build_transform(cfg, train=True, seed=5)
    outputs = [first(DIGIT) for _ in range(4)]
    for out in outputs:
        assert out.dtype == np.float32 and out.shape == (28, 28)
        assert out.min() >= NORMALIZED_BLACK - 1e-5 and out.max() <= NORMALIZED_WHITE + 1e-5
    assert not np.array_equal(outputs[0], outputs[1])  # fresh randomness per call, not one fixed transform
    replay = build_transform(cfg, train=True, seed=5)
    assert all(np.array_equal(a, replay(DIGIT)) for a in outputs)  # same seed -> same stream
    other = build_transform(cfg, train=True, seed=6)
    assert not np.array_equal(outputs[0], other(DIGIT))


def test_geometric_fill_is_black():
    """A white image rotated/shifted must gain black (fill=0) pixels, not reflected white ones."""
    transform = build_transform(AugmentConfig(mode="affine", rotate_degrees=10, translate_px=2), train=True, seed=0)
    white = np.full((28, 28), 255, np.uint8)
    minima = [transform(white).min() for _ in range(10)]
    assert min(minima) == pytest.approx(NORMALIZED_BLACK, abs=1e-5)


def test_none_mode_has_no_randomness_even_for_training():
    transform = build_transform(AugmentConfig(mode="none"), train=True, seed=0)
    assert np.array_equal(transform(DIGIT), transform(DIGIT))


def test_elastic_displacement_scales_with_alpha():
    small = measure_elastic_displacement(8, 4, trials=40)
    large = measure_elastic_displacement(16, 4, trials=40)
    assert large["rms_per_axis_px"] / small["rms_per_axis_px"] == pytest.approx(2.0, rel=0.2)


def test_dataloader_workers_get_distinct_augmentation_streams():
    """Without worker reseeding every worker would replay the same 'random' transforms."""
    from engine import make_loader
    from config import TrainConfig

    images = np.stack([DIGIT] * 4)
    manifest = SequenceManifest(
        "w", "train", np.zeros((8, 1), np.uint8), np.ones((8, 1), bool), np.zeros((8, 1), np.int32), np.zeros(8, np.int16)
    )
    dataset = DigitSumDataset(manifest, images, build_transform(AugmentConfig(mode="affine"), train=True, seed=1))
    loader = make_loader(dataset, 1, torch.device("cpu"), TrainConfig(num_workers=2), seed=0)
    first_items = [batch["images"][0, 0, 0] for batch in loader]
    # DataLoader interleaves workers: items 0 and 1 are the first draws of workers 0 and 1.
    assert not torch.equal(first_items[0], first_items[1])
