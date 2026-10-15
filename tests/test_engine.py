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


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_checkpoint_loads_on_another_device(tmp_path, tiny_cfg):
    model, optimizer, scheduler, batch = trained_objects(tiny_cfg)
    path = save_checkpoint(tmp_path / "ckpt.pt", model, optimizer, scheduler, tiny_cfg, TrainState(), {})
    mps_model, _, _ = load_model_from_checkpoint(path, torch.device("mps"))
    with torch.no_grad():
        cpu_logits = model.eval()(**{k: batch[k] for k in INPUTS})
        mps_logits = mps_model.eval()(**{k: batch[k].to("mps") for k in INPUTS}).cpu()
    assert torch.allclose(cpu_logits, mps_logits, atol=1e-4)


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
    sampler.set_epoch(1)
    assert list(sampler) != epoch0 and sorted(list(sampler)) == list(range(20))
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
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["effective_batch_size"] == 32 and (run_dir / "loss_curves.png").exists()

    # Resume and extend to 4 updates: continues mid-epoch from micro-batch 6.
    from engine import config_for_run

    resumed_cfg, checkpoint = config_for_run(None, ["train.max_steps=4"], str(run_dir / "checkpoints" / "last.pt"))
    run_training(resumed_cfg, torch.device("cpu"), resume_checkpoint=checkpoint, resume_path=str(run_dir / "checkpoints" / "last.pt"))
    resumed = load_checkpoint(run_dir / "checkpoints" / "last.pt")
    assert resumed["train_state"]["global_step"] == 4
    assert resumed["train_state"]["micro_batches_into_epoch"] == 8
