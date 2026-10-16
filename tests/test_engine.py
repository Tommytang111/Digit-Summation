import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import dataset as dataset_module
from conftest import make_synthetic_arrays, tiny_config
from config import config_from_dict, config_to_dict
from dataset import save_manifests, save_source_pools
from engine import (
    EpochShuffleSampler,
    TrainState,
    build_optimizer,
    build_scheduler,
    capture_rng_state,
    load_checkpoint,
    load_model_from_checkpoint,
    restore_rng_state,
    run_training,
    save_checkpoint,
)
from model import build_model
from prepare_data import build_all
from sanity_checks import synthetic_batch

INPUTS = ("images", "token_ids", "is_image", "padding_mask")


def trained_objects(cfg, steps=2):
    torch.manual_seed(0)
    model = build_model(cfg)
    optimizer = build_optimizer(model, 1e-3, 0.01)
    scheduler = build_scheduler(optimizer, cfg.train, total_steps=10)
    batch = synthetic_batch(batch_size=8)
    for _ in range(steps):
        optimizer.zero_grad()
        F.cross_entropy(model(**{k: batch[k] for k in INPUTS}), batch["targets"]).backward()
        optimizer.step()
        scheduler.step()
    return model, optimizer, scheduler, batch


def test_checkpoint_round_trip(tmp_path, tiny_cfg):
    model, optimizer, scheduler, batch = trained_objects(tiny_cfg)
    state = TrainState(epoch=1, global_step=2, micro_batches_into_epoch=3, best_val_accuracy=0.5, best_val_loss=1.0, best_step=2)
    data_ids = {"source_pool_id": "abc", "manifest_ids": {"train": "x"}}
    path = save_checkpoint(tmp_path / "ckpt.pt", model, optimizer, scheduler, tiny_cfg, state, data_ids)

    loaded_model, loaded_cfg, checkpoint = load_model_from_checkpoint(path, torch.device("cpu"))  # weights_only=True
    assert loaded_cfg == tiny_cfg and checkpoint["data_ids"] == data_ids
    assert TrainState(**checkpoint["train_state"]) == state
    model.eval(), loaded_model.eval()
    with torch.no_grad():
        assert torch.equal(model(**{k: batch[k] for k in INPUTS}), loaded_model(**{k: batch[k] for k in INPUTS}))
    fresh_optimizer = build_optimizer(loaded_model, 1e-3, 0.01)
    fresh_optimizer.load_state_dict(checkpoint["optimizer"])
    for old, new in zip(optimizer.state.values(), fresh_optimizer.state.values()):
        assert torch.equal(old["exp_avg"], new["exp_avg"])
    fresh_scheduler = build_scheduler(fresh_optimizer, tiny_cfg.train, total_steps=10)
    fresh_scheduler.load_state_dict(checkpoint["scheduler"])
    assert fresh_scheduler.last_epoch == scheduler.last_epoch


def test_checkpoint_from_before_the_pooling_option_rebuilds_with_max_pooling(tmp_path, tiny_cfg):
    model, optimizer, scheduler, _ = trained_objects(tiny_cfg)
    path = save_checkpoint(tmp_path / "ckpt.pt", model, optimizer, scheduler, tiny_cfg, TrainState(), {})
    payload = torch.load(path, weights_only=True)
    del payload["config"]["model"]["cnn_pooling"]  # what a pre-option checkpoint looks like
    torch.save(payload, path)
    legacy_model, legacy_cfg, _ = load_model_from_checkpoint(path, torch.device("cpu"))
    assert legacy_cfg.model.cnn_pooling == "max"
    assert any(isinstance(m, torch.nn.MaxPool2d) for m in legacy_model.image_encoder.features)


def test_checkpoint_from_before_activation_and_batchnorm_options_rebuilds_original_encoder(tmp_path):
    raw = config_to_dict(tiny_config())
    raw["model"].update(cnn_activation="gelu", cnn_pooling="avg", cnn_batch_norm=False)
    cfg = config_from_dict(raw)
    model, optimizer, scheduler, _ = trained_objects(cfg)
    path = save_checkpoint(tmp_path / "ckpt.pt", model, optimizer, scheduler, cfg, TrainState(), {})
    payload = torch.load(path, weights_only=True)
    for key in ("cnn_activation", "cnn_batch_norm"):
        del payload["config"]["model"][key]  # what a checkpoint from before these options looks like
    torch.save(payload, path)
    legacy_model, legacy_cfg, _ = load_model_from_checkpoint(path, torch.device("cpu"))  # strict weight load
    assert (legacy_cfg.model.cnn_activation, legacy_cfg.model.cnn_batch_norm) == ("gelu", False)
    assert [type(m).__name__ for m in legacy_model.image_encoder.features] == ["Conv2d", "GELU", "AvgPool2d"] * 2


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_checkpoint_loads_on_another_device(tmp_path, tiny_cfg):
    model, optimizer, scheduler, batch = trained_objects(tiny_cfg)
    path = save_checkpoint(tmp_path / "ckpt.pt", model, optimizer, scheduler, tiny_cfg, TrainState(), {})
    mps_model, _, _ = load_model_from_checkpoint(path, torch.device("mps"))
    with torch.no_grad():
        cpu_logits = model.eval()(**{k: batch[k] for k in INPUTS})
        mps_logits = mps_model.eval()(**{k: batch[k].to("mps") for k in INPUTS}).cpu()
    assert torch.allclose(cpu_logits, mps_logits, atol=1e-4)


def test_macro_f1_hand_computed_cases():
    from engine import macro_f1

    targets = np.array([0, 0, 1, 2])
    assert macro_f1(targets.copy(), targets, 91) == 1.0
    # Class 0: TP 1, predicted 1, actual 2 -> F1 2/3. Class 1: TP 1 -> 1. Class 2: never predicted -> 0.
    # Class 5: only predicted -> 0. Classes absent from both are skipped.
    predictions = np.array([0, 5, 1, 5])
    assert macro_f1(predictions, targets, 91) == pytest.approx((2 / 3 + 1 + 0 + 0) / 4)


def test_macro_f1_matches_sklearn():
    metrics = pytest.importorskip("sklearn.metrics")
    from engine import macro_f1

    rng = np.random.default_rng(0)
    targets = rng.integers(20, 70, size=500)  # many classes absent, as in real val/test sets
    predictions = np.where(rng.random(500) < 0.6, targets, rng.integers(15, 75, size=500))
    expected = metrics.f1_score(targets, predictions, average="macro", zero_division=0)
    assert macro_f1(predictions, targets, 91) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_device_side_class_counts_match_numpy(device):
    """The training loop accumulates per-class counts with scatter_add_ on the device."""
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    from engine import macro_f1, macro_f1_from_counts

    rng = np.random.default_rng(1)
    targets_np = rng.integers(0, 91, size=256)
    predictions_np = np.where(rng.random(256) < 0.5, targets_np, rng.integers(0, 91, size=256))
    targets, predictions = torch.from_numpy(targets_np).to(device), torch.from_numpy(predictions_np).to(device)
    hits = (predictions == targets).long()
    counts = [torch.zeros(91, dtype=torch.int64, device=device) for _ in range(3)]
    counts[0].scatter_add_(0, targets, hits)
    counts[1].scatter_add_(0, predictions, torch.ones_like(hits))
    counts[2].scatter_add_(0, targets, torch.ones_like(hits))
    from_device = macro_f1_from_counts(*(c.cpu().numpy() for c in counts))
    assert from_device == pytest.approx(macro_f1(predictions_np, targets_np, 91), abs=1e-12)


def test_rng_state_round_trip():
    import random

    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(1).item())
    restore_rng_state(state)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected


def test_epoch_sampler_is_reproducible_and_resumable():
    sampler = EpochShuffleSampler(20, seed=3)
    sampler.set_epoch(0)
    epoch0 = list(sampler)
    assert all(epoch == 0 for _, epoch in epoch0)  # yields (row, epoch) pairs
    sampler.set_epoch(1)
    rows1 = [row for row, _ in sampler]
    assert rows1 != [row for row, _ in epoch0] and sorted(rows1) == list(range(20))
    assert all(epoch == 1 for _, epoch in sampler)
    sampler.set_epoch(0, skip_items=8)
    assert list(sampler) == epoch0[8:] and len(sampler) == 12


def test_weight_decay_groups_exclude_embeddings_norms_biases(tiny_cfg):
    model = build_model(tiny_cfg)
    decay_group, no_decay_group = build_optimizer(model, 1e-3, 0.1).param_groups
    decayed = {id(p) for p in decay_group["params"]}
    assert id(model.token_embedding.weight) not in decayed
    assert id(model.final_norm.weight) not in decayed and id(model.head.bias) not in decayed
    assert id(model.head.weight) in decayed and id(model.layers[0].self_attn.in_proj_weight) in decayed


@pytest.fixture
def prepared_tmp(tmp_path, monkeypatch):
    """Synthetic MNIST + frozen manifests on disk, with the MNIST loader patched (no download)."""
    arrays = make_synthetic_arrays()
    monkeypatch.setattr(dataset_module, "load_mnist_arrays", lambda root, download=False: arrays)
    manifest_dir = tmp_path / "manifests"
    raw = config_to_dict(tiny_config(str(manifest_dir)))
    raw["logging"]["output_dir"] = str(tmp_path / "runs")
    cfg = config_from_dict(raw)
    pools, manifests = build_all(cfg, arrays)
    save_source_pools(pools, manifest_dir)
    save_manifests(manifests, manifest_dir, extra={})
    return cfg, tmp_path


def test_early_stopping_on_validation_accuracy_and_resume(prepared_tmp):
    from engine import config_for_run

    cfg, _ = prepared_tmp
    raw = config_to_dict(cfg)
    # lr ~ 0 leaves the model near-uniform (accuracy ~0.06-0.1), and min_delta 0.5 makes any later
    # gain too small to count, so only the first validation is an improvement (deterministic).
    raw["train"].update(lr=1e-9, max_steps=20, eval_every_steps=1, log_every_steps=1, early_stopping_patience=2, early_stopping_min_delta=0.5)
    raw["logging"]["run_name"] = "early-stop"
    run_dir = run_training(config_from_dict(raw), torch.device("cpu"))
    last = run_dir / "checkpoints" / "last.pt"

    state = load_checkpoint(last)["train_state"]
    assert state["global_step"] == 3 and state["stopped_early"]  # improve at 1, then 2 misses -> stop at 3
    assert state["validations_without_improvement"] == 2
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["stopped_early"] and summary["final_optimizer_step"] == 3 and summary["planned_optimizer_steps"] == 16  # 2 epochs x 8
    records = [json.loads(line) for line in open(run_dir / "metrics.jsonl")]
    assert [r["val/validations_without_improvement"] for r in records if "val/loss" in r] == [0, 1, 2]

    # Resuming with the same patience stays stopped; a larger patience continues training.
    same_cfg, checkpoint = config_for_run(None, [], str(last))
    run_training(same_cfg, torch.device("cpu"), resume_checkpoint=checkpoint, resume_path=str(last))
    assert load_checkpoint(last)["train_state"]["global_step"] == 3
    longer_cfg, checkpoint = config_for_run(None, ["train.early_stopping_patience=5"], str(last))
    run_training(longer_cfg, torch.device("cpu"), resume_checkpoint=checkpoint, resume_path=str(last))
    resumed = load_checkpoint(last)["train_state"]
    assert resumed["global_step"] == 6 and resumed["validations_without_improvement"] == 5 and resumed["stopped_early"]


def test_early_stopping_disabled_by_default():
    from config import ExperimentConfig

    assert ExperimentConfig().train.early_stopping_patience is None


def test_train_state_ignores_fields_from_other_versions():
    saved = {**asdict_state(TrainState(global_step=7)), "lowest_val_loss": 0.5}  # field of an older version
    assert TrainState.from_checkpoint(saved).global_step == 7


def asdict_state(state):
    from dataclasses import asdict

    return asdict(state)


def test_training_counts_optimizer_steps_with_accumulation_and_resumes(prepared_tmp):
    cfg, tmp_path = prepared_tmp
    raw = config_to_dict(cfg)
    raw["train"].update(grad_accum_steps=2, max_steps=3, eval_every_steps=2, log_every_steps=1)
    raw["logging"]["run_name"] = "accum"
    cfg = config_from_dict(raw)
    run_dir = run_training(cfg, torch.device("cpu"))

    last = load_checkpoint(run_dir / "checkpoints" / "last.pt")
    assert last["train_state"]["global_step"] == 3  # optimizer updates, not micro-batches
    assert last["train_state"]["micro_batches_into_epoch"] == 6  # 3 updates x 2 micro-batches
    records = [json.loads(line) for line in open(run_dir / "metrics.jsonl")]
    assert [r["optimizer_step"] for r in records if "train/loss" in r] == [1, 2, 3]
    assert [r["optimizer_step"] for r in records if "val/loss" in r] == [2, 3]
    assert all(0.0 <= r["train/macro_f1"] <= 1.0 for r in records if "train/loss" in r)
    assert all(0.0 <= r["val/macro_f1"] <= 1.0 for r in records if "val/loss" in r)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["effective_batch_size"] == 32 and (run_dir / "loss_curves.png").exists()
    assert 0.0 <= summary["best_val_macro_f1"] <= 1.0

    # Resume and extend to 4 updates: continues mid-epoch from micro-batch 6.
    from engine import config_for_run

    resumed_cfg, checkpoint = config_for_run(None, ["train.max_steps=4"], str(run_dir / "checkpoints" / "last.pt"))
    run_training(resumed_cfg, torch.device("cpu"), resume_checkpoint=checkpoint, resume_path=str(run_dir / "checkpoints" / "last.pt"))
    resumed = load_checkpoint(run_dir / "checkpoints" / "last.pt")
    assert resumed["train_state"]["global_step"] == 4
    assert resumed["train_state"]["micro_batches_into_epoch"] == 8
