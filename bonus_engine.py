"""Training, free-generation scoring, and checkpoints for the length-generalization bonus.

Reuses the main engine's device, seeding, sampler, loader, optimizer, scheduler, TrainState,
checkpoint, and RNG helpers, plus ExperimentLogger. The main-task loop (engine.run_training) is
not touched.

Training and validation measure two different things:
    train/loss, val/loss      teacher-forced cross-entropy: the decoder is fed the correct
                              previous characters (shifted targets). Averaged over valid tokens.
    val/accuracy              FREE GENERATION: from BOS, the model's own previous predictions,
                              greedy, until EOS or the cap. A row is correct only if the complete
                              answer string is exactly right. This is the reported metric and the
                              checkpoint-selection metric.

Checkpoint selection (recorded in summary.json and every checkpoint) uses validation lengths
bonus.min_train_length..max_train_length only (1..10 by default). Longer validation and test
lengths are never evaluated during training; they are reserved for evaluate_bonus.py.

Gradient accumulation is not supported in the bonus loop (train.grad_accum_steps must be 1),
which keeps the loop short. Resume semantics match the main loop.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import visualization as viz
from augmentations import build_transform, derive_seed
from bonus_data import MultiLengthDataset, bonus_collate, eval_lengths, manifest_name, train_lengths
from bonus_model import bonus_model_inputs, build_bonus_model, position_schemes, teacher_forced_loss
from config import ExperimentConfig, config_from_dict, config_to_dict
from dataset import PreparedData, load_manifests, load_mnist_arrays, load_source_pools
from engine import (
    EpochShuffleSampler,
    TrainState,
    augmentation_seed,
    autocast_context,
    build_optimizer,
    build_scheduler,
    device_report,
    load_checkpoint,
    make_loader,
    restore_rng_state,
    save_checkpoint,
    seed_everything,
    synchronize,
    warn_if_mps_fallback,
)
from experiment_logging import ExperimentLogger, read_metrics
from model import count_parameters
from tokenizer import OUT_PAD_ID, OUTPUT_VOCAB, VOCAB, parse_generated

TASK = "bonus_decoder"


def selection_rule(cfg) -> str:
    b = cfg.bonus
    return (
        f"best aggregate free-generation exact-answer accuracy on validation lengths {b.min_train_length}-{b.max_train_length} "
        "(equal rows per length); ties -> lower teacher-forced validation loss"
    )


# ----------------------------------------------------------------------------- data


def load_bonus_data(cfg) -> PreparedData:
    """MNIST arrays + frozen bonus pools/manifests written by prepare_bonus_data.py."""
    directory = Path(cfg.bonus.manifest_dir)
    if not (directory / "manifests.json").exists():
        raise FileNotFoundError(f"No bonus manifests in {directory}. Run `python prepare_bonus_data.py --config <cfg>` first.")
    return PreparedData(load_mnist_arrays(cfg.data.root), load_source_pools(directory), load_manifests(directory))


def check_bonus_manifests(cfg, data: PreparedData) -> None:
    """The manifests on disk must match the config's lengths and sizes."""
    b = cfg.bonus
    expected = {manifest_name("val", L): b.num_eval_per_length for L in eval_lengths(cfg)}
    expected.update({manifest_name("test", L): b.num_eval_per_length for L in eval_lengths(cfg)})
    expected.update({manifest_name("train", L): None for L in train_lengths(cfg)})
    for name, count in expected.items():
        if name not in data.manifests:
            raise ValueError(f"Bonus manifest {name!r} missing from {b.manifest_dir}; re-run prepare_bonus_data.py with this config")
        if count is not None and data.manifests[name].num_sequences != count:
            raise ValueError(f"{name} has {data.manifests[name].num_sequences} rows, config asks for {count}; re-run prepare_bonus_data.py")
    total_train = sum(data.manifests[manifest_name("train", L)].num_sequences for L in train_lengths(cfg))
    if total_train != b.num_train_sequences:
        raise ValueError(f"Training manifests hold {total_train} rows, config asks for {b.num_train_sequences}")


def training_dataset(cfg, data: PreparedData, augment: bool = True) -> MultiLengthDataset:
    """Training lengths as one dataset. Augmentation is seeded per (epoch, row), separately per length."""
    transform = build_transform(cfg.augment, train=augment)  # shared; reseeded per row from each length's seed
    base = augmentation_seed(cfg)
    return MultiLengthDataset(
        [data.dataset(manifest_name("train", L), transform, augment_seed=derive_seed(base, L) if augment else None) for L in train_lengths(cfg)]
    )


def evaluation_dataset(cfg, data: PreparedData, manifests: list) -> MultiLengthDataset:
    """Validation/test/diagnostic manifests (names or SequenceManifest objects), deterministic preprocessing."""
    transform = build_transform(cfg.augment, train=False)
    return MultiLengthDataset([data.dataset(m, transform) for m in manifests])


def evaluation_loader(cfg, dataset, device):
    return make_loader(dataset, cfg.bonus.eval_batch_size, device, cfg.train, collate_fn=bonus_collate)


# ----------------------------------------------------------------------------- scoring


@torch.no_grad()
def score(model, loader, device: torch.device, cfg, keep_first_step_logits: bool = False) -> dict:
    """Free-generation results per row, plus the teacher-forced loss, in one pass (eval mode).

    Returns numpy arrays (one entry per row, in loader order):
        target, prediction (-1 if invalid), text, status, correct, abs_error (nan if invalid),
        length, image_count, index, tf_loss_sum / tf_tokens (per-row teacher-forced loss parts)
    plus "tf_loss" (mean over valid target tokens) and optionally "first_step_logits" [N, 13]
    (the logits after BOS, used by the structural diagnostics).
    """
    was_training = model.training
    model.eval()
    rows = {key: [] for key in ("target", "prediction", "text", "status", "correct", "abs_error", "length", "image_count", "index", "tf_loss_sum", "tf_tokens")}
    first_step = []
    loss_sum, token_count = 0.0, 0
    for batch in loader:
        inputs = bonus_model_inputs(batch, device)
        targets = batch["decoder_target"].to(device)
        with autocast_context(device, cfg.train.amp_dtype):
            memory = model.encode(inputs["images"], inputs["token_ids"], inputs["is_image"], inputs["padding_mask"])
            logits = model.decode(memory, inputs["padding_mask"], inputs["decoder_input"], inputs["decoder_padding_mask"])
            generated = model.generate_from_memory(memory, inputs["padding_mask"], cfg.bonus.max_new_tokens)
        valid = targets != OUT_PAD_ID  # [B, T]
        per_token = F.cross_entropy(logits.float().transpose(1, 2), targets, ignore_index=OUT_PAD_ID, reduction="none")  # [B, T], 0 at pads
        rows["tf_loss_sum"].extend(per_token.sum(dim=1).cpu().tolist())
        rows["tf_tokens"].extend(valid.sum(dim=1).cpu().tolist())
        loss_sum += float(per_token.sum())
        token_count += int(valid.sum())
        if keep_first_step_logits:
            first_step.append(logits[:, 0].float().cpu())
        for row, generated_ids in enumerate(generated.cpu().tolist()):
            text, status = parse_generated(generated_ids)
            target = int(batch["targets"][row])
            prediction = int(text) if text is not None else -1
            rows["target"].append(target)
            rows["prediction"].append(prediction)
            rows["text"].append(text if text is not None else "")
            rows["status"].append(status)
            rows["correct"].append(status == "valid" and prediction == target)
            rows["abs_error"].append(abs(prediction - target) if status == "valid" else math.nan)
        rows["length"].extend(batch["lengths"].tolist())
        rows["image_count"].extend((batch["is_image"] & ~batch["padding_mask"]).sum(dim=1).tolist())
        rows["index"].extend(batch["indices"].tolist())
    model.train(was_training)
    result = {key: np.array(values) for key, values in rows.items()}
    result["tf_loss"] = loss_sum / max(1, token_count)
    if keep_first_step_logits:
        result["first_step_logits"] = torch.cat(first_step).numpy()
    return result


def summarize(result: dict, mask: np.ndarray | None = None) -> dict:
    """Accuracy, invalid/missing-EOS rates, status counts, and MAE on valid outputs (with its denominator)."""
    keep = np.ones(len(result["target"]), dtype=bool) if mask is None else mask
    n = int(keep.sum())
    if n == 0:
        return {"n": 0, "accuracy": None, "invalid_rate": None, "missing_eos_rate": None, "n_valid": 0, "mae_valid": None, "status_counts": {}}
    status = result["status"][keep]
    valid = status == "valid"
    errors = result["abs_error"][keep][valid]
    return {
        "n": n,
        "accuracy": float(result["correct"][keep].mean()),
        "invalid_rate": float((~valid).mean()),
        "missing_eos_rate": float((status == "missing_eos").mean()),
        "n_valid": int(valid.sum()),
        "mae_valid": float(errors.mean()) if errors.size else None,  # denominator = n_valid
        "status_counts": {s: int((status == s).sum()) for s in sorted(set(status.tolist()))},
    }


def teacher_forced_loss_of(result: dict, mask: np.ndarray) -> float | None:
    """Teacher-forced loss over the selected rows, averaged over their valid target tokens."""
    tokens = result["tf_tokens"][mask].sum()
    return float(result["tf_loss_sum"][mask].sum() / tokens) if tokens else None


def by_length(result: dict, mask: np.ndarray | None = None) -> list[dict]:
    keep = np.ones(len(result["target"]), dtype=bool) if mask is None else mask
    rows = []
    for L in sorted(set(result["length"][keep].tolist())):
        selected = keep & (result["length"] == L)
        rows.append({"group": int(L), **summarize(result, selected), "tf_loss": teacher_forced_loss_of(result, selected)})
    return rows


# ----------------------------------------------------------------------------- checkpoints


def checkpoint_extra(cfg) -> dict:
    return {
        "task": TASK,
        "output_vocab": dict(OUTPUT_VOCAB),
        "architecture": cfg.bonus.architecture,
        "positions": position_schemes(cfg.bonus.architecture),
        "selection_rule": selection_rule(cfg),
    }


def load_bonus_model(path: Path, device: torch.device):
    """Rebuild a bonus model from its checkpoint (any device). Returns (model, cfg, checkpoint)."""
    checkpoint = load_checkpoint(path)
    if checkpoint.get("task") != TASK:
        raise ValueError(f"{path} is not a bonus checkpoint (task={checkpoint.get('task', 'classifier')!r}); use evaluate.py")
    if checkpoint["vocab"] != dict(VOCAB) or checkpoint["output_vocab"] != dict(OUTPUT_VOCAB):
        raise ValueError("Checkpoint vocabularies differ from tokenizer.VOCAB / OUTPUT_VOCAB")
    cfg = config_from_dict(checkpoint["config"])
    model = build_bonus_model(cfg)
    model.load_state_dict(checkpoint["model"])
    return model.to(device), cfg, checkpoint


# ----------------------------------------------------------------------------- training


def run_bonus_training(cfg: ExperimentConfig, device: torch.device, resume_checkpoint: dict | None = None, resume_path: str | None = None) -> Path:
    """Teacher-forced training with periodic free-generation validation on lengths min..max train."""
    if cfg.train.grad_accum_steps != 1:
        raise ValueError("The bonus loop does not support gradient accumulation; set train.grad_accum_steps=1")
    warn_if_mps_fallback(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = cfg.train.allow_tf32
        torch.backends.cudnn.allow_tf32 = cfg.train.allow_tf32
    autocast_context(device, cfg.train.amp_dtype)  # fail fast on unsupported combinations

    data = load_bonus_data(cfg)
    check_bonus_manifests(cfg, data)
    data_ids = data.data_ids()
    if resume_checkpoint is not None and resume_checkpoint["data_ids"] != data_ids:
        raise ValueError("Resume checkpoint was trained on different bonus pools/manifests than are on disk")

    seed_everything(cfg.train.seed)
    if resume_path:
        run_dir = Path(resume_path).resolve().parent.parent
    else:
        run_name = cfg.logging.run_name or f"{cfg.name}-{cfg.bonus.architecture}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run_dir = Path(cfg.logging.output_dir) / run_name
        if run_dir.exists():
            raise FileExistsError(f"{run_dir} already exists; choose another logging.run_name")
    checkpoint_dir = run_dir / "checkpoints"

    train_set = training_dataset(cfg, data)
    sampler = EpochShuffleSampler(len(train_set), seed=derive_seed(cfg.train.seed, 2))
    train_loader = make_loader(train_set, cfg.train.batch_size, device, cfg.train, sampler=sampler, drop_last=True, collate_fn=bonus_collate)
    val_names = [manifest_name("val", L) for L in train_lengths(cfg)]  # in-distribution only
    val_loader = evaluation_loader(cfg, evaluation_dataset(cfg, data, val_names), device)

    model = build_bonus_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    steps_per_epoch = len(train_set) // cfg.train.batch_size
    if steps_per_epoch == 0:
        raise ValueError("Training manifests are smaller than one batch")
    total_steps = cfg.train.epochs * steps_per_epoch
    if cfg.train.max_steps is not None:
        total_steps = min(total_steps, cfg.train.max_steps)
    scheduler = build_scheduler(optimizer, cfg.train, total_steps)
    state = TrainState()
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        state = TrainState.from_checkpoint(resume_checkpoint["train_state"])
        restore_rng_state(resume_checkpoint["rng"])
        print(f"Resumed from {resume_path} at optimizer step {state.global_step} (epoch {state.epoch}).")

    parameters = count_parameters(model)
    run_config = {
        **config_to_dict(cfg),
        "resolved": {
            "task": TASK,
            "device": device_report(device),
            "parameters": parameters,
            "data_ids": data_ids,
            "train_rows_per_length": {L: data.manifests[manifest_name("train", L)].num_sequences for L in train_lengths(cfg)},
            "optimizer_steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": total_steps,
            "selection_rule": selection_rule(cfg),
            "validation_lengths_used_in_training": train_lengths(cfg),
        },
    }
    logger = ExperimentLogger(run_dir, run_config, cfg.logging, resume=resume_checkpoint is not None)
    if resume_checkpoint is not None:
        logger.truncate_after(state.global_step)
    print(f"Run directory: {run_dir}")
    print(f"Device: {device_report(device)}")
    print(f"Architecture {cfg.bonus.architecture} | parameters {parameters['total']:,} | train rows {len(train_set):,} over lengths {train_lengths(cfg)}")
    print(f"{steps_per_epoch} optimizer steps/epoch, planned {total_steps}; selection: {selection_rule(cfg)}")

    patience = cfg.train.early_stopping_patience
    extra = checkpoint_extra(cfg)

    def validate_and_checkpoint() -> bool:
        result = score(model, val_loader, device, cfg)
        overall = summarize(result)
        accuracy, loss = overall["accuracy"], result["tf_loss"]
        if accuracy > state.early_stopping_best_accuracy + cfg.train.early_stopping_min_delta:
            state.early_stopping_best_accuracy, state.validations_without_improvement = accuracy, 0
        else:
            state.validations_without_improvement += 1
        stop = patience is not None and state.validations_without_improvement >= patience
        state.stopped_early = state.stopped_early or stop
        improved = accuracy > state.best_val_accuracy or (accuracy == state.best_val_accuracy and loss < state.best_val_loss)
        if improved:
            state.best_val_accuracy, state.best_val_loss, state.best_step = accuracy, loss, state.global_step
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, scheduler, cfg, state, data_ids, extra=extra)
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, scheduler, cfg, state, data_ids, extra=extra)
        metrics = {
            "epoch": round(state.epoch + state.micro_batches_into_epoch / steps_per_epoch, 4),
            "val/loss": loss,  # teacher-forced
            "val/accuracy": accuracy,  # free generation, lengths min..max train
            "val/invalid_rate": overall["invalid_rate"],
            "val/missing_eos_rate": overall["missing_eos_rate"],
            "val/best_accuracy": state.best_val_accuracy,
            "val/validations_without_improvement": state.validations_without_improvement,
            **{f"val/accuracy_by_length/L{row['group']:02d}": row["accuracy"] for row in by_length(result)},
        }
        logger.log_metrics(metrics, step=state.global_step)
        print(
            f"  [val 1-{cfg.bonus.max_train_length}] step {state.global_step:>6}  tf-loss {loss:.4f}  free-gen acc {accuracy:.4f}  "
            f"invalid {overall['invalid_rate']:.3f}{' *best*' if improved else ''}"
        )
        if stop:
            print(f"Early stopping at step {state.global_step}: val accuracy has not improved for {state.validations_without_improvement} validations.")
        return stop

    def interval_reset():
        synchronize(device)
        return {"loss_sum": torch.zeros((), device=device), "token_correct": torch.zeros((), device=device), "tokens": torch.zeros((), device=device),
                "samples": 0, "steps": 0, "start": time.perf_counter()}

    last_validated = state.global_step if resume_checkpoint is not None else -1
    done = state.global_step >= total_steps or (patience is not None and state.validations_without_improvement >= patience)
    if not done:
        state.stopped_early = False
    interval = interval_reset()
    started = time.perf_counter()
    model.train()
    try:
        while not done and state.epoch < cfg.train.epochs:
            skip = state.micro_batches_into_epoch
            sampler.set_epoch(state.epoch, skip_items=skip * cfg.train.batch_size)
            for micro_index, batch in enumerate(train_loader, start=skip):
                inputs = bonus_model_inputs(batch, device)
                targets = batch["decoder_target"].to(device, non_blocking=True)
                with autocast_context(device, cfg.train.amp_dtype):
                    logits = model(**inputs)  # [B, T, 13], teacher forced
                loss = teacher_forced_loss(logits, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.train.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
                optimizer.step()
                scheduler.step()
                state.global_step += 1
                state.micro_batches_into_epoch = micro_index + 1
                valid = targets != OUT_PAD_ID
                interval["loss_sum"] += loss.detach() * valid.sum()
                interval["token_correct"] += ((logits.argmax(dim=-1) == targets) & valid).sum()
                interval["tokens"] += valid.sum()
                interval["samples"] += targets.shape[0]
                interval["steps"] += 1

                if state.global_step % cfg.train.log_every_steps == 0:
                    synchronize(device)
                    elapsed = time.perf_counter() - interval["start"]
                    tokens = interval["tokens"].item()
                    mean_loss = interval["loss_sum"].item() / tokens
                    if not math.isfinite(mean_loss):
                        raise FloatingPointError(f"Non-finite training loss at optimizer step {state.global_step}")
                    logger.log_metrics(
                        {
                            "epoch": round(state.epoch + state.micro_batches_into_epoch / steps_per_epoch, 4),
                            "train/loss": mean_loss,  # teacher-forced, per valid token
                            "train/tf_token_accuracy": interval["token_correct"].item() / tokens,  # teacher-forced, NOT answer accuracy
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/samples_per_sec": interval["samples"] / elapsed,
                        },
                        step=state.global_step,
                    )
                    print(f"step {state.global_step:>6}/{total_steps}  tf-loss {mean_loss:.4f}  tf-token-acc {interval['token_correct'].item() / tokens:.3f}  {interval['samples'] / elapsed:.0f} seq/s")
                    interval = interval_reset()

                if state.global_step % cfg.train.eval_every_steps == 0:
                    should_stop = validate_and_checkpoint()
                    last_validated = state.global_step
                    interval = interval_reset()
                    if should_stop:
                        done = True
                        break
                if state.global_step >= total_steps:
                    done = True
                    break
            if not done or state.micro_batches_into_epoch >= steps_per_epoch:
                state.epoch += 1
                state.micro_batches_into_epoch = 0
    except NotImplementedError as error:
        if device.type == "mps":
            raise RuntimeError("An operation is not implemented on MPS; run with --device cpu (PYTORCH_ENABLE_MPS_FALLBACK is not enabled).") from error
        raise

    if last_validated != state.global_step:
        validate_and_checkpoint()
    curves = viz.plot_loss_curves(read_metrics(logger.metrics_path), run_dir / "loss_curves.png", f"{run_dir.name}: teacher-forced loss and free-generation val accuracy")
    logger.log_image("curves/loss", curves, step=state.global_step)
    logger.log_checkpoint_artifact(checkpoint_dir / "best.pt", name=f"{run_dir.name}-best")
    summary = {
        "run_dir": str(run_dir),
        "task": TASK,
        "config_name": cfg.name,
        "architecture": cfg.bonus.architecture,
        "positions": checkpoint_extra(cfg)["positions"],
        "augment_mode": cfg.augment.mode,
        "seed": cfg.train.seed,
        "device": str(device),
        "parameters": parameters["total"],
        "final_optimizer_step": state.global_step,
        "planned_optimizer_steps": total_steps,
        "stopped_early": state.stopped_early,
        "epochs_completed": state.epoch,
        "best_step": state.best_step,
        "best_val_accuracy": state.best_val_accuracy,
        "best_val_loss": state.best_val_loss,
        "selection_rule": selection_rule(cfg),
        "effective_batch_size": cfg.train.batch_size,
        "training_seconds_this_session": round(time.perf_counter() - started, 1),
        "data_ids": data_ids,
        "train_state": asdict(state),
        "note": f"Lengths {cfg.bonus.max_train_length + 1}-{cfg.bonus.max_eval_length} were not evaluated during training. Use evaluate_bonus.py.",
    }
    logger.log_summary(summary)
    logger.finish()
    print(f"Done. best val (lengths {cfg.bonus.min_train_length}-{cfg.bonus.max_train_length}) free-gen accuracy {state.best_val_accuracy:.4f} at step {state.best_step}.")
    return run_dir
