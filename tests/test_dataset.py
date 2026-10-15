import numpy as np
import pytest
import torch

from augmentations import build_transform
from config import AugmentConfig
from dataset import (
    DigitSumDataset,
    SPLITS,
    build_source_pools,
    collate_batch,
    compute_targets,
    generate_manifest,
    input_keys,
)
from prepare_data import build_all, run_leakage_audit
from tokenizer import PAD_ID

FULL = {"train": 1.0, "val": 1.0, "test": 1.0}


def eval_transform():
    return build_transform(AugmentConfig(mode="none"), train=False)


# ----------------------------------------------------------------------------- source pools


def test_pools_disjoint_stratified_and_test_preserved(synthetic_arrays):
    pools = build_source_pools(synthetic_arrays, val_source_size=50, fractions=FULL, seed=0)
    assert len(pools.ids["val"]) == 50 and len(pools.ids["train"]) == 250
    assert not set(pools.ids["train"]) & set(pools.ids["val"])
    assert np.array_equal(pools.ids["test"], np.arange(100))  # official test pool untouched
    labels = synthetic_arrays.labels["train"]
    assert np.bincount(labels[pools.ids["val"]], minlength=10).tolist() == [5] * 10  # 30 per digit -> 5 held out


def test_exact_duplicates_resolved_before_generation(synthetic_arrays):
    images, labels = synthetic_arrays.images, synthetic_arrays.labels
    images["train"][5] = images["test"][0]  # dev copy of a test image -> removed from dev
    labels["train"][11] = labels["train"][10]
    images["train"][11] = images["train"][10]  # same-label dev duplicate -> keep lowest index
    other = (labels["train"][20] + 1) % 10
    labels["train"][21] = other
    images["train"][21] = images["train"][20]  # conflicting labels -> drop both
    synthetic_arrays._hashes.clear()
    pools = build_source_pools(synthetic_arrays, val_source_size=50, fractions=FULL, seed=0)
    report = pools.report
    assert report["dev_images_duplicating_test_removed"] == 1
    assert report["dev_internal_duplicates_removed"] == 1
    assert report["dev_conflicting_label_images_removed"] == 2
    dev = set(pools.ids["train"]) | set(pools.ids["val"])
    assert 5 not in dev and 10 in dev and 11 not in dev and 20 not in dev and 21 not in dev
    assert report["dev_pool_after_dedup"] == 300 - 4


def test_source_fractions_are_nested_and_stratified(synthetic_arrays):
    small = build_source_pools(synthetic_arrays, 50, {"train": 0.2, "val": 0.4, "test": 0.5}, seed=0)
    large = build_source_pools(synthetic_arrays, 50, {"train": 0.6, "val": 1.0, "test": 1.0}, seed=0)
    for split in SPLITS:
        assert set(small.ids[split]) <= set(large.ids[split])
    labels = synthetic_arrays.labels["train"]
    assert np.bincount(labels[small.ids["train"]], minlength=10).tolist() == [5] * 10  # round(0.2 * 25)


# ----------------------------------------------------------------------------- manifests


def make_manifest(arrays, pools, seed=1, p=0.5, n=200, pool_split="train", **kw):
    return generate_manifest("m", pool_split, pools, arrays, n, 10, p, seed, **kw)


def test_manifest_targets_sources_and_reproducibility(synthetic_arrays):
    pools = build_source_pools(synthetic_arrays, 50, FULL, seed=0)
    manifest = make_manifest(synthetic_arrays, pools)
    assert manifest.digits.dtype == np.uint8 and manifest.is_image.dtype == bool
    assert manifest.source_index.dtype == np.int32 and manifest.digits.shape == (200, 10)
    labels = synthetic_arrays.labels["train"]
    assert np.array_equal(labels[manifest.source_index[manifest.is_image]], manifest.digits[manifest.is_image])
    assert set(manifest.source_index[manifest.is_image]) <= set(pools.ids["train"])
    assert (manifest.source_index[~manifest.is_image] == -1).all()
    assert np.array_equal(manifest.targets, manifest.digits.astype(int).sum(axis=1))
    assert make_manifest(synthetic_arrays, pools).manifest_id() == manifest.manifest_id()
    assert make_manifest(synthetic_arrays, pools, seed=2).manifest_id() != manifest.manifest_id()


def test_targets_come_from_mnist_labels(synthetic_arrays):
    pools = build_source_pools(synthetic_arrays, 50, FULL, seed=0)
    manifest = make_manifest(synthetic_arrays, pools, p=1.0)
    corrupted = manifest.digits.copy()
    corrupted[0, 0] = (corrupted[0, 0] + 1) % 10  # label metadata no longer matches the image
    with pytest.raises(AssertionError):
        compute_targets(corrupted, manifest.is_image, manifest.source_index, synthetic_arrays.labels["train"])


def test_modality_extremes(synthetic_arrays):
    pools = build_source_pools(synthetic_arrays, 50, FULL, seed=0)
    assert not make_manifest(synthetic_arrays, pools, p=0.0).is_image.any()
    assert make_manifest(synthetic_arrays, pools, p=1.0).is_image.all()


def test_cross_split_exclusion_resamples_colliding_rows(synthetic_arrays):
    pools = build_source_pools(synthetic_arrays, 50, FULL, seed=0)
    # Length-1 all-text sequences have only 10 possible inputs ("t0".."t9"): block half of them.
    blocked_keys = {f"t{digit}" for digit in range(5)}
    allowed = generate_manifest("b", "train", pools, synthetic_arrays, 50, 1, 0.0, seed=4, exclude_canonical=blocked_keys)
    _, allowed_keys = input_keys(allowed, synthetic_arrays.hashes("train"))
    assert not set(allowed_keys) & blocked_keys
    assert set(allowed.digits.ravel().tolist()) <= {5, 6, 7, 8, 9}
    assert allowed.metadata["rows_resampled_to_avoid_cross_split_duplicates"] > 0
    # Blocking every possible input cannot be satisfied: the generator must fail loudly.
    with pytest.raises(RuntimeError):
        generate_manifest("c", "train", pools, synthetic_arrays, 5, 1, 0.0, seed=4, exclude_canonical={f"t{d}" for d in range(10)}, max_rounds=20)


def test_oracle_and_permutation_views(synthetic_arrays):
    pools = build_source_pools(synthetic_arrays, 50, FULL, seed=0)
    manifest = make_manifest(synthetic_arrays, pools)
    oracle = manifest.with_all_text()
    assert not oracle.is_image.any() and np.array_equal(oracle.targets, manifest.targets)
    permuted = manifest.permuted(seed=0)
    assert np.array_equal(np.sort(permuted.digits, axis=1), np.sort(manifest.digits, axis=1))
    assert np.array_equal(permuted.targets, manifest.targets)
    assert not np.array_equal(permuted.digits, manifest.digits)


# ----------------------------------------------------------------------------- dataset + collate


def test_dataset_item_contract(synthetic_arrays):
    pools = build_source_pools(synthetic_arrays, 50, FULL, seed=0)
    manifest = make_manifest(synthetic_arrays, pools)
    dataset = DigitSumDataset(manifest, synthetic_arrays.images["train"], eval_transform())
    for index in range(20):
        item = dataset[index]
        assert item["images"].shape == (10, 1, 28, 28) and item["images"].dtype == torch.float32
        assert item["token_ids"].dtype == torch.int64 and item["is_image"].dtype == torch.bool
        assert torch.isfinite(item["images"]).all()
        assert (item["images"][~item["is_image"]] == 0.0).all()  # finite placeholder at text slots
        assert (item["token_ids"][item["is_image"]] == PAD_ID).all()  # images never carry label tokens
        text_digits = item["digits"][~item["is_image"]]
        assert torch.equal(item["token_ids"][~item["is_image"]], text_digits + 1)  # "d" -> d + 1
        assert item["target"] == int(manifest.targets[index])


def test_zero_digit_text_slot_is_not_padding(synthetic_arrays):
    pools = build_source_pools(synthetic_arrays, 50, FULL, seed=0)
    manifest = make_manifest(synthetic_arrays, pools, p=0.0, n=50)
    manifest.digits[0] = 0
    dataset = DigitSumDataset(manifest, synthetic_arrays.images["train"], eval_transform())
    batch = collate_batch([dataset[0]])
    assert (batch["token_ids"][0] == 1).all()
    assert not batch["padding_mask"].any()


def _item(length, image_positions):
    is_image = torch.zeros(length, dtype=torch.bool)
    is_image[list(image_positions)] = True
    token_ids = torch.where(is_image, torch.tensor(PAD_ID), torch.arange(length) % 10 + 1)
    return {
        "images": torch.randn(length, 1, 28, 28) * is_image[:, None, None, None],
        "token_ids": token_ids,
        "is_image": is_image,
        "target": 7,
        "digits": torch.arange(length) % 10,
        "index": length,
    }


def test_collate_pads_variable_lengths_with_a_separate_mask():
    batch = collate_batch([_item(3, [0]), _item(5, [1, 4])])
    assert batch["images"].shape == (2, 5, 1, 28, 28) and batch["targets"].dtype == torch.int64
    assert batch["padding_mask"].tolist() == [[False, False, False, True, True], [False] * 5]
    assert batch["token_ids"][0, 3:].tolist() == [PAD_ID, PAD_ID]
    assert not batch["is_image"][0, 3:].any()
    assert batch["digits"][0, 3:].tolist() == [-1, -1]
    # An image slot also has PAD_ID as its token, yet it is not padding:
    assert batch["token_ids"][0, 0] == PAD_ID and not batch["padding_mask"][0, 0]
    assert all(v.dtype != torch.float64 for v in batch.values())


# ----------------------------------------------------------------------------- audit


def test_audit_passes_on_generated_data(synthetic_arrays, tiny_cfg):
    pools, manifests = build_all(tiny_cfg, synthetic_arrays)
    report = run_leakage_audit(synthetic_arrays, pools, manifests)
    assert report["passed"], [c for c in report["checks"] if not c["passed"]]


def test_audit_detects_injected_leaks(synthetic_arrays, tiny_cfg):
    pools, manifests = build_all(tiny_cfg, synthetic_arrays)
    train, test = manifests["train"], manifests["test"]
    # 1. A training image operand that comes from the validation pool.
    row, col = np.argwhere(train.is_image)[0]
    val_image = pools.ids["val"][labels_equal(synthetic_arrays, pools, train.digits[row, col])]
    train.source_index[row, col] = val_image
    # 2. A test row duplicated as all-text in train, and a permuted copy in val.
    for manifest in (train, test):
        manifest.digits[1] = np.arange(10)
        manifest.is_image[1] = False
        manifest.source_index[1] = -1
        manifest.targets[1] = 45
    val = manifests["val"]
    val.digits[1], val.is_image[1], val.source_index[1], val.targets[1] = np.arange(10)[::-1], False, -1, 45
    failed = {c["check"] for c in run_leakage_audit(synthetic_arrays, pools, manifests)["checks"] if not c["passed"]}
    assert "train: image IDs inside its 'train' pool" in failed
    assert "no exact complete-input duplicates: test vs train" in failed
    assert "no permutation-equivalent inputs: test vs val" in failed
    assert "no exact complete-input duplicates: test vs val" not in failed  # reordered, not identical


def labels_equal(arrays, pools, digit):
    return int(np.flatnonzero(arrays.labels["train"][pools.ids["val"]] == digit)[0])
