import json
import math

import numpy as np
import pytest
import torch

import bonus_engine
from bonus_data import build_bonus_data
from bonus_engine import checkpoint_extra, load_bonus_model, run_bonus_training, summarize
from bonus_model import ARCHITECTURES, build_bonus_model, position_schemes
from config import config_from_dict, config_to_dict
from conftest import make_synthetic_arrays, tiny_config
from dataset import save_manifests, save_source_pools
from engine import TrainState, build_optimizer, build_scheduler, load_checkpoint, load_model_from_checkpoint, save_checkpoint
from model import build_model
from tokenizer import OUTPUT_VOCAB


def bonus_cfg(tmp_path, **train):
    raw = config_to_dict(tiny_config())
    raw["bonus"].update(manifest_dir=str(tmp_path / "bonus"), min_train_length=1, max_train_length=3, max_eval_length=5,
                        num_train_sequences=96, num_eval_per_length=8, num_all_image_per_length=4, eval_batch_size=32)
    raw["train"].update(batch_size=16, max_steps=4, eval_every_steps=2, log_every_steps=1, **train)
    raw["logging"].update(output_dir=str(tmp_path / "runs"))
    return config_from_dict(raw)


@pytest.fixture
def prepared_bonus(tmp_path, monkeypatch):
    arrays = make_synthetic_arrays()
    monkeypatch.setattr(bonus_engine, "load_mnist_arrays", lambda root, download=False: arrays)
    cfg = bonus_cfg(tmp_path)
    pools, manifests = build_bonus_data(cfg, arrays)
    save_source_pools(pools, tmp_path / "bonus")
    save_manifests(manifests, tmp_path / "bonus", extra={})
    return cfg, tmp_path


def test_summarize_counts_invalid_outputs_and_mae_denominator():
    result = {
        "target": np.array([5, 12, 30, 7]),
        "correct": np.array([True, False, False, False]),
        "status": np.array(["valid", "valid", "missing_eos", "empty"]),
        "abs_error": np.array([0.0, 3.0, math.nan, math.nan]),
    }
    s = summarize(result)
    assert s["n"] == 4 and s["accuracy"] == 0.25 and s["invalid_rate"] == 0.5 and s["missing_eos_rate"] == 0.25
    assert s["n_valid"] == 2 and s["mae_valid"] == 1.5  # MAE only over the 2 valid outputs
    assert s["status_counts"] == {"empty": 1, "missing_eos": 1, "valid": 2}


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_bonus_checkpoint_round_trip_and_cross_task_guards(tmp_path, architecture):
    cfg = bonus_cfg(tmp_path)
    cfg.bonus.architecture = architecture
    torch.manual_seed(0)
    model = build_bonus_model(cfg)
    optimizer = build_optimizer(model, 1e-3, 0.0)
    scheduler = build_scheduler(optimizer, cfg.train, 10)
    path = save_checkpoint(tmp_path / "bonus.pt", model, optimizer, scheduler, cfg, TrainState(global_step=3), {"source_pool_id": "x", "manifest_ids": {}}, extra=checkpoint_extra(cfg))
    loaded, loaded_cfg, checkpoint = load_bonus_model(path, torch.device("cpu"))
    assert checkpoint["task"] == "bonus_decoder" and checkpoint["output_vocab"] == OUTPUT_VOCAB
    assert checkpoint["architecture"] == architecture and checkpoint["positions"] == position_schemes(architecture)
    assert loaded_cfg.bonus.architecture == architecture and checkpoint["train_state"]["global_step"] == 3
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), loaded.state_dict().values()))
    with pytest.raises(ValueError, match="not a main-task classifier"):
        load_model_from_checkpoint(path, torch.device("cpu"))
    main = build_model(cfg)
    main_path = save_checkpoint(tmp_path / "main.pt", main, build_optimizer(main, 1e-3, 0.0), None, cfg, TrainState(), {})
    with pytest.raises(ValueError, match="not a bonus checkpoint"):
        load_bonus_model(main_path, torch.device("cpu"))
    if torch.backends.mps.is_available():
        on_mps, _, _ = load_bonus_model(path, torch.device("mps"))
        assert next(on_mps.parameters()).device.type == "mps"


@pytest.mark.parametrize("architecture", ("items_only_sinusoidal", "items_only_rope", "items_only_alibi"))
def test_training_runs_selects_on_trained_lengths_and_resumes(prepared_bonus, architecture):
    cfg, tmp_path = prepared_bonus
    cfg.bonus.architecture = architecture
    run_dir = run_bonus_training(cfg, torch.device("cpu"))
    last = load_checkpoint(run_dir / "checkpoints" / "last.pt")
    assert last["task"] == "bonus_decoder" and last["train_state"]["global_step"] == 4
    records = [json.loads(line) for line in open(run_dir / "metrics.jsonl")]
    val_keys = {k for r in records for k in r if k.startswith("val/accuracy_by_length/")}
    assert val_keys == {"val/accuracy_by_length/L01", "val/accuracy_by_length/L02", "val/accuracy_by_length/L03"}  # never 4-5
    assert [r["optimizer_step"] for r in records if "val/loss" in r] == [2, 4]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert "validation lengths 1-3" in summary["selection_rule"] and summary["architecture"] == architecture

    from engine import config_for_run

    resumed_cfg, checkpoint = config_for_run(None, ["train.max_steps=6"], str(run_dir / "checkpoints" / "last.pt"))
    run_bonus_training(resumed_cfg, torch.device("cpu"), resume_checkpoint=checkpoint, resume_path=str(run_dir / "checkpoints" / "last.pt"))
    assert load_checkpoint(run_dir / "checkpoints" / "last.pt")["train_state"]["global_step"] == 6


def test_gradient_accumulation_is_rejected(prepared_bonus):
    cfg, _ = prepared_bonus
    cfg.train.grad_accum_steps = 2
    with pytest.raises(ValueError, match="gradient accumulation"):
        run_bonus_training(cfg, torch.device("cpu"))
