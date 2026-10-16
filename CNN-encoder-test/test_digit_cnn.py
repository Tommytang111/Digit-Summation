"""Fast synthetic tests (CPU, no download):  ../.torch/bin/python -m pytest CNN-encoder-test"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from digit_cnn import DigitCNN, DigitImageDataset, digit_report, implied_sum_report

from augmentations import build_transform
from config import AugmentConfig, ExperimentConfig
from dataset import SequenceManifest, compute_targets
from engine import macro_f1
from model import build_model


def test_encoder_is_identical_to_the_sum_models_image_encoder():
    cfg = ExperimentConfig()
    reference = build_model(cfg).image_encoder.state_dict()
    candidate = DigitCNN(cfg.model).encoder.state_dict()
    assert list(candidate) == list(reference)
    assert all(candidate[k].shape == reference[k].shape for k in reference)
    assert DigitCNN(cfg.model)(torch.randn(3, 1, 28, 28)).shape == (3, 10)


def test_default_encoder_is_relu_maxpool_batchnorm():
    model = DigitCNN(ExperimentConfig().model)
    layer_types = [type(m).__name__ for m in model.encoder.features]
    assert layer_types == ["Conv2d", "BatchNorm2d", "ReLU", "MaxPool2d"] * 2
    assert all(m.bias is None for m in model.encoder.features if isinstance(m, torch.nn.Conv2d))  # BN replaces bias
    model.train()
    assert model(torch.randn(4, 1, 28, 28)).shape == (4, 10)
    model.eval()  # running statistics in eval mode
    assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)


def test_original_encoder_still_available_through_config():
    cfg = ExperimentConfig()
    cfg.model.cnn_activation, cfg.model.cnn_pooling, cfg.model.cnn_batch_norm = "gelu", "avg", False
    layer_types = [type(m).__name__ for m in DigitCNN(cfg.model).encoder.features]
    assert layer_types == ["Conv2d", "GELU", "AvgPool2d"] * 2


def test_cnn_uses_the_configured_pooling():
    cfg = ExperimentConfig()
    cfg.model.cnn_pooling = "max"
    assert any(isinstance(m, torch.nn.MaxPool2d) for m in DigitCNN(cfg.model).encoder.features)


def test_dataset_items_and_per_epoch_seeding():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(20, 28, 28), dtype=np.uint8)
    labels = rng.integers(0, 10, size=20)
    dataset = DigitImageDataset(images, labels, np.arange(5, 15), build_transform(AugmentConfig(mode="affine"), train=True), augment_seed=3)
    pixels, label, image_id = dataset[(0, 0)]
    assert pixels.shape == (1, 28, 28) and pixels.dtype == torch.float32
    assert image_id == 5 and label == int(labels[5])
    assert torch.equal(dataset[(0, 0)][0], pixels)  # same (epoch, image) -> same augmentation
    assert not torch.equal(dataset[(0, 1)][0], pixels)  # next epoch -> new augmentation


def test_digit_report_counts_and_macro_f1():
    labels = np.array([0, 0, 1, 2, 2, 2])
    predictions = np.array([0, 1, 1, 2, 2, 0])
    report = digit_report(predictions, labels)
    assert report["accuracy"] == pytest.approx(4 / 6) and report["errors"] == 2
    assert report["macro_f1"] == pytest.approx(macro_f1(predictions, labels, 10))
    assert report["confusion"][0][1] == 1 and report["confusion"][2][0] == 1
    assert report["per_digit"][2]["recall"] == pytest.approx(2 / 3) and report["per_digit"][2]["support"] == 3


def test_implied_sum_ceiling_uses_cnn_digits_for_images_only():
    labels = np.array([3, 7, 5, 9])  # partition labels by image ID
    digits = np.array([[3, 1, 7], [5, 2, 2], [4, 4, 4]], dtype=np.uint8)
    is_image = np.array([[True, False, True], [True, False, False], [False, False, False]])
    source = np.where(is_image, np.array([[0, -1, 1], [2, -1, -1], [-1, -1, -1]]), -1).astype(np.int32)
    targets = compute_targets(digits, is_image, source, labels).astype(np.int16)
    manifest = SequenceManifest("toy", "val", digits, is_image, source, targets)

    perfect = implied_sum_report(manifest, np.array([3, 7, 5, 9]), image_accuracy=1.0)
    assert perfect["exact_sum_accuracy"] == 1.0
    one_wrong = implied_sum_report(manifest, np.array([3, 7, 6, 9]), image_accuracy=0.75)  # image 2 read as 6
    assert one_wrong["exact_sum_accuracy"] == pytest.approx(2 / 3)  # only row 1 changes; text-only row 2 unaffected
    with pytest.raises(ValueError):
        implied_sum_report(manifest, np.array([3, -1, 5, 9]), image_accuracy=1.0)  # unpredicted image
