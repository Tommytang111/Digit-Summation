import numpy as np
import pytest
import torch

from augmentations import build_transform
from bonus_data import MultiLengthDataset, bonus_collate, build_bonus_data, constructed_manifest, manifest_name, manifest_plan
from config import AugmentConfig, config_from_dict, config_to_dict
from conftest import tiny_config
from dataset import DigitSumDataset, compute_targets
from prepare_data import run_leakage_audit
from tokenizer import BOS_ID, EOS_ID, OUT_PAD_ID, OUTPUT_VOCAB, PAD_ID


def bonus_cfg(**bonus):
    raw = config_to_dict(tiny_config())
    raw["bonus"].update(min_train_length=1, max_train_length=3, max_eval_length=5, num_train_sequences=61, num_eval_per_length=6, num_all_image_per_length=3, **bonus)
    return config_from_dict(raw)


def test_plan_uses_uniform_training_lengths_and_every_eval_length():
    plan = {spec["name"]: spec for spec in manifest_plan(bonus_cfg())}
    assert sorted(spec["count"] for name, spec in plan.items() if name.startswith("train")) == [20, 20, 21]  # 61 split evenly
    assert {manifest_name("val", L) for L in range(1, 6)} <= set(plan) and manifest_name("test", 5, "all_image") in plan
    assert all(plan[manifest_name("test", L, "all_image")]["p"] == 1.0 for L in range(1, 6))
    assert len({spec["seed"] for spec in plan.values()}) == len(plan)  # every manifest has its own recorded seed


def test_generated_manifests_have_right_lengths_targets_and_pools(synthetic_arrays):
    cfg = bonus_cfg()
    pools, manifests = build_bonus_data(cfg, synthetic_arrays)
    for name, m in manifests.items():
        expected_length = int(name[-2:])
        assert m.sequence_length == expected_length
        labels = synthetic_arrays.labels[m.partition]
        assert np.array_equal(compute_targets(m.digits, m.is_image, m.source_index, labels), m.targets.astype(int))
        assert set(m.source_index[m.is_image].tolist()) <= set(pools.ids[m.pool_split].tolist())
    assert manifests[manifest_name("test", 4, "all_image")].is_image.all()
    assert not any(n.startswith("train") and int(n[-2:]) > 3 for n in manifests)  # no long training rows


def test_audit_report_policy_passes_clean_data_and_catches_cross_pool_leaks(synthetic_arrays):
    cfg = bonus_cfg()
    pools, manifests = build_bonus_data(cfg, synthetic_arrays)
    report = run_leakage_audit(synthetic_arrays, pools, manifests, text_only_duplicates="report")
    assert report["passed"], [c for c in report["checks"] if not c["passed"]]
    # Inject a leak: the first operand of a training row becomes a VALIDATION-pool image.
    train = manifests[manifest_name("train", 2)]
    val_image = int(pools.ids["val"][0])
    train.is_image[0, 0], train.source_index[0, 0] = True, val_image
    train.digits[0, 0] = synthetic_arrays.labels["train"][val_image]  # keep targets consistent, so only the pool check fires
    train.targets[0] = compute_targets(train.digits[:1], train.is_image[:1], train.source_index[:1], synthetic_arrays.labels["train"])[0]
    failed = [c["check"] for c in run_leakage_audit(synthetic_arrays, pools, manifests, text_only_duplicates="report")["checks"] if not c["passed"]]
    assert any("train_L02" in c and "inside its 'train' pool" in c for c in failed)


def test_multi_length_dataset_maps_rows_and_seeds_augmentation_per_epoch(synthetic_arrays):
    cfg = bonus_cfg()
    _, manifests = build_bonus_data(cfg, synthetic_arrays)
    transform = build_transform(AugmentConfig(mode="affine"), train=True)
    subsets = [DigitSumDataset(manifests[manifest_name("train", L)], synthetic_arrays.images["train"], transform, augment_seed=100 + L) for L in (1, 2, 3)]
    dataset = MultiLengthDataset(subsets)
    assert len(dataset) == 61 and dataset.locate(0) == (0, 0) and dataset.locate(len(subsets[0])) == (1, 0)
    assert np.bincount(dataset.row_lengths).tolist() == [0, 21, 20, 20]  # uniform over lengths
    item = dataset[(30, 0)]
    assert item["length"] == 2 and item["index"] == 30 and item["token_ids"].shape == (2,)
    image_row = next(r for r in range(len(dataset)) if dataset.row_lengths[r] == 3 and manifests["train_L03"].is_image[dataset.locate(r)[1]].any())
    assert torch.equal(dataset[(image_row, 4)]["images"], dataset[(image_row, 4)]["images"])
    assert not torch.equal(dataset[(image_row, 4)]["images"], dataset[(image_row, 5)]["images"])


def test_bonus_collate_variable_lengths_masks_and_answer_tensors(synthetic_arrays):
    cfg = bonus_cfg()
    pools, _ = build_bonus_data(cfg, synthetic_arrays)
    rows = [constructed_manifest(f"c{L}", "val", [digits], [[False] * L], synthetic_arrays, pools, seed=1) for L, digits in ((1, [0]), (4, [9, 9, 9, 9]), (2, [5, 0]))]
    transform = build_transform(AugmentConfig(mode="none"), train=False)
    items = [DigitSumDataset(m, synthetic_arrays.images["train"], transform)[0] for m in rows]
    batch = bonus_collate(items)
    assert batch["padding_mask"].tolist() == [[False, True, True, True], [False] * 4, [False, False, True, True]]
    assert batch["token_ids"][0].tolist() == [1, PAD_ID, PAD_ID, PAD_ID]  # the zero operand is token 1, padding is 0
    assert batch["lengths"].tolist() == [1, 4, 2]
    digit = lambda c: OUTPUT_VOCAB[c]  # noqa: E731
    assert batch["decoder_input"].tolist() == [[BOS_ID, digit("0"), OUT_PAD_ID], [BOS_ID, digit("3"), digit("6")], [BOS_ID, digit("5"), OUT_PAD_ID]]
    assert batch["decoder_target"].tolist() == [[digit("0"), EOS_ID, OUT_PAD_ID], [digit("3"), digit("6"), EOS_ID], [digit("5"), EOS_ID, OUT_PAD_ID]]
    assert batch["decoder_padding_mask"].tolist() == [[False, False, True], [False, False, False], [False, False, True]]


def test_constructed_manifest_uses_labels_and_pool_images(synthetic_arrays):
    cfg = bonus_cfg()
    pools, _ = build_bonus_data(cfg, synthetic_arrays)
    m = constructed_manifest("repeat", "test", [[3, 3, 3], [0, 0, 0]], [[True] * 3, [True, False, True]], synthetic_arrays, pools, seed=2)
    assert m.targets.tolist() == [9, 0]
    assert (synthetic_arrays.labels["test"][m.source_index[m.is_image]] == m.digits[m.is_image]).all()
    with pytest.raises(ValueError):
        constructed_manifest("bad", "test", [[1], [1, 2]], [[False], [False, False]], synthetic_arrays, pools, seed=0)
