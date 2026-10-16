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


def test_children_share_one_stream_instead_of_identical_copies():
    """Compose.set_random_seed seeds every child identically; seed_compose shares one stream."""
    transform = build_transform(AugmentConfig(mode="affine_elastic"), train=True, seed=3)
    affine, elastic = transform.compose.transforms[0], transform.compose.transforms[1]
    assert affine.random_generator is elastic.random_generator and affine.py_random is elastic.py_random
    import albumentations as A

    naive = A.Compose([A.Affine(p=1.0), A.ElasticTransform(p=1.0)], seed=3)
    first, second = naive.transforms
    assert first.random_generator.random() == second.random_generator.random()  # the duplication avoided


def _row_dataset(rows=24, length=3, mode="affine_elastic"):
    images = np.random.default_rng(1).integers(0, 256, size=(5, 28, 28), dtype=np.uint8)
    manifest = SequenceManifest(
        "rows",
        "train",
        np.zeros((rows, length), np.uint8),
        np.ones((rows, length), bool),
        np.random.default_rng(2).integers(0, 5, size=(rows, length)).astype(np.int32),
        np.zeros(rows, np.int16),
    )
    return DigitSumDataset(manifest, images, build_transform(AugmentConfig(mode=mode, elastic_p=1.0), train=True), augment_seed=11)


def test_row_augmentation_is_a_pure_function_of_seed_epoch_row():
    dataset = _row_dataset()
    reference = dataset[(5, 0)]["images"]
    dataset[(7, 0)], dataset[(5, 3)]  # other accesses in between must not matter
    assert torch.equal(dataset[(5, 0)]["images"], reference)
    assert not torch.equal(dataset[(5, 1)]["images"], reference)  # new epoch -> new augmentation
    first_slot = dataset[(5, 0)]["images"][0]
    other_row = _row_dataset()
    other_row.manifest.source_index[6] = other_row.manifest.source_index[5]
    assert not torch.equal(other_row[(6, 0)]["images"][0], first_slot)  # same pixels, other row -> other draw


def test_batches_identical_for_any_num_workers_and_persistence():
    """The property that makes a num_workers speed comparison controlled: same data, every epoch."""
    from config import TrainConfig
    from engine import EpochShuffleSampler, make_loader

    def epochs_of_batches(num_workers, persistent):
        dataset = _row_dataset()
        sampler = EpochShuffleSampler(len(dataset), seed=4)
        cfg = TrainConfig(num_workers=num_workers, persistent_workers=persistent)
        loader = make_loader(dataset, 4, torch.device("cpu"), cfg, sampler=sampler, drop_last=True)
        out = []
        for epoch in range(2):
            sampler.set_epoch(epoch)
            out.append([(batch["indices"].clone(), batch["images"].clone()) for batch in loader])
        return out

    reference = epochs_of_batches(0, False)
    assert not torch.equal(reference[0][0][1], reference[1][0][1])  # epochs differ
    for num_workers, persistent in ((2, False), (2, True)):
        other = epochs_of_batches(num_workers, persistent)
        for ref_epoch, other_epoch in zip(reference, other):
            for (ref_idx, ref_img), (idx, img) in zip(ref_epoch, other_epoch):
                assert torch.equal(ref_idx, idx) and torch.equal(ref_img, img), (num_workers, persistent)
