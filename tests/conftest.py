"""Synthetic fixtures so the test suite never needs to download MNIST."""

from __future__ import annotations

import numpy as np
import pytest

from config import ExperimentConfig, config_from_dict, config_to_dict
from dataset import MnistArrays


def make_synthetic_arrays(train_per_digit: int = 30, test_per_digit: int = 10, seed: int = 0) -> MnistArrays:
    """Random-pixel 'MNIST': every image unique, labels balanced and shuffled."""
    rng = np.random.default_rng(seed)

    def partition(per_digit: int):
        labels = np.repeat(np.arange(10), per_digit)
        rng.shuffle(labels)
        images = rng.integers(0, 256, size=(len(labels), 28, 28), dtype=np.uint8)
        return images, labels.astype(np.int64)

    train_images, train_labels = partition(train_per_digit)
    test_images, test_labels = partition(test_per_digit)
    return MnistArrays(images={"train": train_images, "test": test_images}, labels={"train": train_labels, "test": test_labels})


@pytest.fixture
def synthetic_arrays() -> MnistArrays:
    return make_synthetic_arrays()


def tiny_config(manifest_dir: str = "unused", **data_overrides) -> ExperimentConfig:
    """Small config matching the synthetic arrays (300 train/val images, 100 test images)."""
    raw = config_to_dict(ExperimentConfig())
    raw["data"].update(
        manifest_dir=manifest_dir,
        val_source_size=50,
        num_train_sequences=128,
        num_val_sequences=32,
        num_test_sequences=32,
        num_diagnostic_sequences=16,
        **data_overrides,
    )
    raw["train"].update(device="cpu", batch_size=16, epochs=2, eval_every_steps=4, log_every_steps=2)
    raw["eval"].update(batch_size=32, num_gallery_examples=4, num_stress_image_sequences=4)
    return config_from_dict(raw)


@pytest.fixture
def tiny_cfg() -> ExperimentConfig:
    return tiny_config()
