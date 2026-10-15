"""Shared training / evaluation / checkpoint logic used by train_local.py and train_cloud.py.

Vocabulary used throughout:
    micro-batch     one forward/backward pass on `train.batch_size` sequences
    optimizer step  one optimizer.step() = `grad_accum_steps` micro-batches; `global_step` counts
                    these and is the x-axis of every plot
    epoch           one pass over the (finite, fixed) training manifest; with drop_last, an epoch
                    has floor(N / batch_size) micro-batches

Reproducibility:
- Python, NumPy, and torch are seeded from train.seed. The shuffle order of epoch e comes from
  a permutation seeded by (seed, e). The Albumentations RNG is reseeded per epoch (num_workers=0)
  or per worker per epoch (num_workers>0). Augmentation streams therefore differ between
  num_workers settings.
- Resume is exact at epoch boundaries for data order and model/optimizer/scheduler/RNG state.
  Mid-epoch resume skips already-consumed batches of the same order, but the augmentation draws
  for the remaining batches differ from the uninterrupted run (the stream restarts at the
  epoch's seed). GPU/MPS kernels may also be nondeterministic.
"""

from __future__ import annotations

import contextlib
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

import visualization as viz
from augmentations import build_transform, seed_worker
from config import ExperimentConfig, apply_override, config_from_dict, config_to_dict, load_config
from dataset import PreparedData, collate_batch, load_prepared_data
from experiment_logging import ExperimentLogger, read_metrics
from model import DigitSumTransformer, build_model, count_parameters, model_inputs
from tokenizer import VOCAB, number_to_string

CHECKPOINT_FORMAT = 1


# ----------------------------------------------------------------------------- devices


def resolve_device(requested: str, allowed: tuple[str, ...] = ("mps", "cuda", "cpu")) -> torch.device:
    """'auto' picks the first available of `allowed` in order; explicit requests must be available."""
    available = {
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
        "cpu": True,
    }
    if requested == "auto":
        chosen = next(name for name in allowed if available[name])
    else:
        chosen = requested
    if chosen not in allowed:
        raise ValueError(f"Device {chosen!r} is not allowed by this entry point (allowed: {allowed})")
    if not available[chosen]:
        detail = f" (MPS built: {torch.backends.mps.is_built()})" if chosen == "mps" else ""
        raise RuntimeError(f"Device {chosen!r} was requested but is not available{detail}")
    return torch.device(chosen)


def mps_fallback_enabled() -> bool:
    return os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0").strip().lower() in ("1", "true", "yes")


def warn_if_mps_fallback(device: torch.device) -> None:
    """Make CPU fallback loud: with it, unsupported MPS ops silently run on the CPU."""
    if device.type == "mps" and mps_fallback_enabled():
        print(
            "\n" + "!" * 78 + "\n"
            "WARNING: PYTORCH_ENABLE_MPS_FALLBACK=1 is set. Unsupported MPS ops will run on the CPU,\n"
            "which hides missing kernels and can be very slow. This project does not need it;\n"
            "unset it (`unset PYTORCH_ENABLE_MPS_FALLBACK`) to surface unsupported ops as errors.\n" + "!" * 78 + "\n"
        )


def device_report(device: torch.device) -> dict:
    report = {"device": str(device), "torch": torch.__version__, "platform": platform.platform()}
    if device.type == "cuda":
        report.update(
            gpu=torch.cuda.get_device_name(device),
            capability=".".join(map(str, torch.cuda.get_device_capability(device))),
            cuda_runtime=torch.version.cuda,
            bf16_supported=torch.cuda.is_bf16_supported(),
        )
    elif device.type == "mps":
        report.update(gpu="Apple MPS", mps_fallback_env=mps_fallback_enabled())
    else:
        report.update(cpu=platform.processor() or platform.machine(), threads=torch.get_num_threads())
    return report


def synchronize(device: torch.device) -> None:
    """Wait for queued accelerator work, so wall-clock timings measure finished computation."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def autocast_context(device: torch.device, amp_dtype: str):
    """bf16 autocast on CUDA only (checked); otherwise plain float32."""
    if amp_dtype == "none":
        return contextlib.nullcontext()
    if amp_dtype == "bf16":
        if device.type != "cuda":
            raise ValueError("train.amp_dtype=bf16 is only supported on CUDA in this project")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support bfloat16; set train.amp_dtype=none")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"Unknown amp_dtype {amp_dtype!r}")


# ----------------------------------------------------------------------------- seeding + loaders


def derive_seed(*parts: int) -> int:
    """Independent 32-bit seed for a named purpose, e.g. derive_seed(seed, 3, epoch)."""
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators


class EpochShuffleSampler(Sampler[int]):
    """Shuffles with a permutation seeded by (seed, epoch); can skip items to resume mid-epoch."""

    def __init__(self, num_items: int, seed: int):
        self.num_items = num_items
        self.seed = seed
        self.epoch = 0
        self.skip_items = 0

    def set_epoch(self, epoch: int, skip_items: int = 0) -> None:
        self.epoch = epoch
        self.skip_items = skip_items

    def __iter__(self):
        order = np.random.default_rng([self.seed, self.epoch]).permutation(self.num_items)
        return iter(order[self.skip_items :].tolist())

    def __len__(self) -> int:
        return self.num_items - self.skip_items


def make_loader(dataset, batch_size: int, device: torch.device, train_cfg, sampler=None, drop_last=False, seed=0) -> DataLoader:
    """DataLoader with the project's collate, worker seeding, and CUDA-only pinned memory."""
    use_workers = train_cfg.num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,  # order comes from the sampler (or is sequential for evaluation)
        drop_last=drop_last,
        collate_fn=collate_batch,
        num_workers=train_cfg.num_workers,
        pin_memory=bool(train_cfg.pin_memory and device.type == "cuda"),
        worker_init_fn=seed_worker if use_workers else None,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=False,  # workers are re-created (and reseeded) every epoch
    )


# ----------------------------------------------------------------------------- optimizer


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW; weight decay only on >=2-D weights of Linear/Conv/attention (not biases, norms, embeddings)."""
    decay, no_decay = [], []
    for module in model.modules():
        for _, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            if parameter.ndim >= 2 and not isinstance(module, torch.nn.Embedding):
                decay.append(parameter)
            else:
                no_decay.append(parameter)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr)


def build_scheduler(optimizer, train_cfg, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Multiplier on the base LR per optimizer step: optional linear warmup, then constant or cosine."""
    warmup = train_cfg.warmup_steps

    def multiplier(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if train_cfg.lr_schedule == "warmup_cosine":
            progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


# ----------------------------------------------------------------------------- evaluation


@dataclass
class EvalOutput:
    loss: float
    accuracy: float
    mae: float
    num_examples: int
    predictions: np.ndarray  # int64 [N]
    targets: np.ndarray  # int64 [N]
    indices: np.ndarray  # int64 [N] manifest rows
    logits: np.ndarray | None = None  # float32 [N, C] when requested

    def metrics(self) -> dict:
        return {"loss": self.loss, "accuracy": self.accuracy, "mae": self.mae, "n": self.num_examples}


@torch.no_grad()
def evaluate(model, loader, device: torch.device, amp_dtype: str = "none", keep_logits: bool = False) -> EvalOutput:
    """Full pass in eval mode (dropout off). MAE is |predicted sum - true sum| averaged."""
    was_training = model.training
    model.eval()
    loss_sum = torch.zeros((), device=device)
    predictions, targets, indices, logits_list = [], [], [], []
    for batch in loader:
        inputs = model_inputs(batch, device)
        batch_targets = batch["targets"].to(device)
        with autocast_context(device, amp_dtype):
            logits = model(**inputs)
        logits = logits.float()
        loss_sum += F.cross_entropy(logits, batch_targets, reduction="sum")
        predictions.append(logits.argmax(dim=-1).cpu())
        targets.append(batch["targets"])
        indices.append(batch["indices"])
        if keep_logits:
            logits_list.append(logits.cpu())
    model.train(was_training)
    predictions = torch.cat(predictions).numpy().astype(np.int64)
    targets = torch.cat(targets).numpy().astype(np.int64)
    count = len(targets)
    return EvalOutput(
        loss=float(loss_sum.item()) / count,
        accuracy=float((predictions == targets).mean()),
        mae=float(np.abs(predictions - targets).mean()),
        num_examples=count,
        predictions=predictions,
        targets=targets,
        indices=torch.cat(indices).numpy().astype(np.int64),
        logits=torch.cat(logits_list).numpy() if keep_logits else None,
    )


# ----------------------------------------------------------------------------- checkpoints


@dataclass
class TrainState:
    epoch: int = 0  # epochs fully completed
    global_step: int = 0  # optimizer.step() calls so far
    micro_batches_into_epoch: int = 0  # consumed micro-batches of the current epoch
    best_val_accuracy: float = -1.0
    best_val_loss: float = math.inf
    best_step: int = -1


def capture_rng_state() -> dict:
    """All RNG states as tensors/plain types so torch.load(weights_only=True) can read them."""
    numpy_state = np.random.get_state()
    python_version, python_internal, python_gauss = random.getstate()
    state = {
        "python": [python_version, list(python_internal), python_gauss],
        "numpy": {
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict) -> None:
    version, internal, gauss = state["python"]
    random.setstate((version, tuple(internal), gauss))
    numpy_state = state["numpy"]
    np.random.set_state(
        ("MT19937", numpy_state["keys"].numpy().astype(np.uint32), numpy_state["pos"], numpy_state["has_gauss"], numpy_state["cached_gaussian"])
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def save_checkpoint(path: Path, model, optimizer, scheduler, cfg: ExperimentConfig, state: TrainState, data_ids: dict) -> Path:
    """Atomic save (write to a temp file, then rename) of everything needed to resume or evaluate."""
    import albumentations
    import torchvision

    payload = {
        "format": CHECKPOINT_FORMAT,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": None,  # bf16 autocast and float32 need no GradScaler
        "config": config_to_dict(cfg),
        "vocab": dict(VOCAB),
        "data_ids": data_ids,
        "train_state": asdict(state),
        "rng": capture_rng_state(),
        # str(): torch.__version__ is a TorchVersion object, which the weights-only unpickler rejects.
        "versions": {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__), "albumentations": str(albumentations.__version__)},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_checkpoint(path: Path) -> dict:
    """Load onto the CPU with the safe weights-only unpickler. Move tensors to the target device afterward."""
    return torch.load(path, map_location="cpu", weights_only=True)


def load_model_from_checkpoint(path: Path, device: torch.device) -> tuple[DigitSumTransformer, ExperimentConfig, dict]:
    """Rebuild the model from the checkpoint's own config and load weights on any device."""
    checkpoint = load_checkpoint(path)
    if checkpoint["vocab"] != dict(VOCAB):
        raise ValueError("Checkpoint vocabulary differs from tokenizer.VOCAB")
    cfg = config_from_dict(checkpoint["config"])
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model"])
    return model.to(device), cfg, checkpoint


def config_for_run(config_path: str | None, overrides: list[str], resume_path: str | None) -> tuple[ExperimentConfig, dict | None]:
    """Fresh run: YAML + overrides. Resume: the checkpoint's own config + overrides (e.g. a larger max_steps)."""
    if resume_path:
        checkpoint = load_checkpoint(Path(resume_path))
        raw = checkpoint["config"]
        for override in overrides:
            apply_override(raw, override)
        return config_from_dict(raw), checkpoint
    if not config_path:
        raise ValueError("--config is required unless --resume is given")
    return load_config(config_path, overrides), None


# ----------------------------------------------------------------------------- training


def check_manifests_match_config(cfg: ExperimentConfig, data: PreparedData) -> None:
    expected = {"train": cfg.data.num_train_sequences, "val": cfg.data.num_val_sequences, "test": cfg.data.num_test_sequences}
    for name, count in expected.items():
        actual = data.manifests[name].num_sequences
        if actual != count:
            raise ValueError(
                f"Manifest {name!r} in {cfg.data.manifest_dir} has {actual} rows but the config asks for {count}. "
                "Re-run prepare_data.py with this config (or point data.manifest_dir elsewhere)."
            )


def _gallery_indices(count: int, k: int, seed: int) -> np.ndarray:
    return np.sort(np.random.default_rng(seed).choice(count, size=min(k, count), replace=False))


def save_training_examples(cfg, data: PreparedData, path: Path, num_rows: int = 8) -> Path:
    """Training-example visualization: what the model sees (augmented, normalized) + targets.

    Uses its own transform instance, so drawing the figure does not perturb the training RNG.
    """
    from augmentations import NORMALIZED_BLACK, NORMALIZED_WHITE

    manifest = data.manifests["train"]
    preview_dataset = data.dataset("train", build_transform(cfg.augment, train=True, seed=derive_seed(cfg.train.seed, 99)))
    rows = []
    for index in _gallery_indices(manifest.num_sequences, num_rows, seed=derive_seed(cfg.train.seed, 98)):
        item = preview_dataset[int(index)]
        rows.append(
            {
                "images": [item["images"][p, 0].numpy() if item["is_image"][p] else None for p in range(len(item["is_image"]))],
                "tokens": [None if item["is_image"][p] else number_to_string(int(item["digits"][p])) for p in range(len(item["is_image"]))],
                "target": item["target"],
                "caption": f"row {int(index)}",
            }
        )
    title = f"Training examples as the model sees them (augment={cfg.augment.mode}, normalized pixels)"
    return viz.plot_sequence_gallery(rows, path, title, pixel_range=(NORMALIZED_BLACK, NORMALIZED_WHITE))


def run_training(cfg: ExperimentConfig, device: torch.device, resume_checkpoint: dict | None = None, resume_path: str | None = None) -> Path:
    """Train with periodic validation; save best/last checkpoints, local metrics, and plots.

    Returns the run directory. Test data is never touched here; use evaluate.py explicitly.
    """
    warn_if_mps_fallback(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = cfg.train.allow_tf32
        torch.backends.cudnn.allow_tf32 = cfg.train.allow_tf32
    autocast_context(device, cfg.train.amp_dtype)  # fail fast on an unsupported combination
    if cfg.train.num_workers == 0 and cfg.train.pin_memory and device.type != "cuda":
        print("Note: train.pin_memory is ignored off CUDA.")

    data = load_prepared_data(cfg.data)
    check_manifests_match_config(cfg, data)
    data_ids = data.data_ids()
    if resume_checkpoint is not None and resume_checkpoint["data_ids"] != data_ids:
        raise ValueError("Resume checkpoint was trained on different source pools/manifests than are on disk")

    seed_everything(cfg.train.seed)
    if resume_path:
        run_dir = Path(resume_path).resolve().parent.parent
    else:
        run_name = cfg.logging.run_name or f"{cfg.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run_dir = Path(cfg.logging.output_dir) / run_name
        if run_dir.exists():
            raise FileExistsError(f"{run_dir} already exists; choose another logging.run_name")
    checkpoint_dir = run_dir / "checkpoints"

    train_transform = build_transform(cfg.augment, train=True, seed=derive_seed(cfg.train.seed, 1))
    eval_transform = build_transform(cfg.augment, train=False)
    train_dataset = data.dataset("train", train_transform)
    val_dataset = data.dataset("val", eval_transform)
    sampler = EpochShuffleSampler(len(train_dataset), seed=derive_seed(cfg.train.seed, 2))
    train_loader = make_loader(train_dataset, cfg.train.batch_size, device, cfg.train, sampler=sampler, drop_last=True)
    val_loader = make_loader(val_dataset, cfg.eval.batch_size, device, cfg.train)

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    accumulation = cfg.train.grad_accum_steps
    micro_batches_per_epoch = len(train_dataset) // cfg.train.batch_size
    steps_per_epoch = micro_batches_per_epoch // accumulation
    if steps_per_epoch == 0:
        raise ValueError("Training manifest is smaller than one effective batch")
    total_steps = cfg.train.epochs * steps_per_epoch
    if cfg.train.max_steps is not None:
        total_steps = min(total_steps, cfg.train.max_steps)
    scheduler = build_scheduler(optimizer, cfg.train, total_steps)
    state = TrainState()

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])  # moves state to parameter devices
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        state = TrainState(**resume_checkpoint["train_state"])
        restore_rng_state(resume_checkpoint["rng"])
        print(f"Resumed from {resume_path} at optimizer step {state.global_step} (epoch {state.epoch}).")

    parameter_counts = count_parameters(model)
    run_config = {
        **config_to_dict(cfg),
        "resolved": {
            "device": device_report(device),
            "parameters": parameter_counts,
            "data_ids": data_ids,
            "source_counts": data.pools.report["final_counts"],
            "sequence_counts": {name: m.num_sequences for name, m in data.manifests.items()},
            "micro_batches_per_epoch": micro_batches_per_epoch,
            "optimizer_steps_per_epoch": steps_per_epoch,
            "effective_batch_size": cfg.train.batch_size * accumulation,
            "total_optimizer_steps": total_steps,
            "augmentation_seed": derive_seed(cfg.train.seed, 1),
            "shuffle_seed": derive_seed(cfg.train.seed, 2),
        },
    }
    logger = ExperimentLogger(run_dir, run_config, cfg.logging, resume=resume_checkpoint is not None)
    if resume_checkpoint is not None:
        logger.truncate_after(state.global_step)
    print(f"Run directory: {run_dir}")
    print(f"Device: {device_report(device)}")
    print(f"Parameters: {parameter_counts['total']:,}  |  source images {data.pools.report['final_counts']}")
    print(
        f"Train sequences {len(train_dataset):,}, val {len(val_dataset):,} | {micro_batches_per_epoch} micro-batches/epoch, "
        f"{steps_per_epoch} optimizer steps/epoch, effective batch {cfg.train.batch_size * accumulation}, "
        f"planned optimizer steps {total_steps}"
    )

    if resume_checkpoint is None:
        examples_path = save_training_examples(cfg, data, run_dir / "train_examples.png")
        logger.log_image("examples/train", examples_path, step=0)

    def validate_and_checkpoint() -> None:
        result = evaluate(model, val_loader, device, cfg.train.amp_dtype)
        improved = result.accuracy > state.best_val_accuracy or (
            result.accuracy == state.best_val_accuracy and result.loss < state.best_val_loss
        )
        if improved:
            state.best_val_accuracy, state.best_val_loss, state.best_step = result.accuracy, result.loss, state.global_step
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, scheduler, cfg, state, data_ids)
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, scheduler, cfg, state, data_ids)
        epoch_progress = state.epoch + state.micro_batches_into_epoch / micro_batches_per_epoch
        logger.log_metrics(
            {
                "epoch": round(epoch_progress, 4),
                "val/loss": result.loss,
                "val/accuracy": result.accuracy,
                "val/mae": result.mae,
                "val/best_accuracy": state.best_val_accuracy,
            },
            step=state.global_step,
        )
        marker = " *best*" if improved else ""
        print(f"  [val] step {state.global_step:>6}  loss {result.loss:.4f}  acc {result.accuracy:.4f}  mae {result.mae:.3f}{marker}")

    def interval_reset():
        synchronize(device)
        return {
            "loss_sum": torch.zeros((), device=device),
            "correct": torch.zeros((), device=device, dtype=torch.int64),
            "grad_norm_sum": torch.zeros((), device=device),
            "samples": 0,
            "steps": 0,
            "start": time.perf_counter(),
        }

    last_validated_step = state.global_step if resume_checkpoint is not None else -1
    interval = interval_reset()
    training_start = time.perf_counter()
    done = state.global_step >= total_steps
    model.train()
    try:
        while not done and state.epoch < cfg.train.epochs:
            skip = state.micro_batches_into_epoch
            sampler.set_epoch(state.epoch, skip_items=skip * cfg.train.batch_size)
            epoch_seed = derive_seed(cfg.train.seed, 3, state.epoch)
            if cfg.train.num_workers == 0:
                train_transform.set_random_seed(epoch_seed)
            else:
                train_loader.generator.manual_seed(epoch_seed)  # -> worker seeds -> seed_worker
            optimizer.zero_grad(set_to_none=True)

            for micro_index, batch in enumerate(train_loader, start=skip):
                inputs = model_inputs(batch, device)
                targets = batch["targets"].to(device, non_blocking=True)
                with autocast_context(device, cfg.train.amp_dtype):
                    logits = model(**inputs)  # [B, 91]
                loss = F.cross_entropy(logits.float(), targets)  # directly on logits
                (loss / accumulation).backward()
                interval["loss_sum"] += loss.detach()
                interval["correct"] += (logits.argmax(dim=-1) == targets).sum()
                interval["samples"] += targets.numel()
                state.micro_batches_into_epoch = micro_index + 1

                if (micro_index + 1) % accumulation != 0:
                    continue  # keep accumulating; no optimizer step yet
                if cfg.train.grad_clip_norm is not None:
                    interval["grad_norm_sum"] += torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state.global_step += 1  # counts optimizer updates only
                interval["steps"] += 1

                if state.global_step % cfg.train.log_every_steps == 0:
                    synchronize(device)
                    elapsed = time.perf_counter() - interval["start"]
                    micro_batches = interval["samples"] / cfg.train.batch_size
                    mean_loss = interval["loss_sum"].item() / micro_batches
                    if not math.isfinite(mean_loss):
                        raise FloatingPointError(f"Non-finite training loss at optimizer step {state.global_step}")
                    metrics = {
                        "epoch": round(state.epoch + state.micro_batches_into_epoch / micro_batches_per_epoch, 4),
                        "train/loss": mean_loss,
                        "train/accuracy": interval["correct"].item() / interval["samples"],
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/samples_per_sec": interval["samples"] / elapsed,  # end-to-end incl. data loading
                        "train/steps_per_sec": interval["steps"] / elapsed,
                    }
                    if cfg.train.grad_clip_norm is not None:
                        metrics["train/grad_norm_preclip"] = interval["grad_norm_sum"].item() / interval["steps"]
                    logger.log_metrics(metrics, step=state.global_step)
                    print(
                        f"step {state.global_step:>6}/{total_steps}  epoch {metrics['epoch']:.2f}  loss {mean_loss:.4f}  "
                        f"acc {metrics['train/accuracy']:.3f}  {metrics['train/samples_per_sec']:.0f} seq/s"
                    )
                    interval = interval_reset()

                if state.global_step % cfg.train.eval_every_steps == 0:
                    validate_and_checkpoint()
                    last_validated_step = state.global_step
                    interval = interval_reset()  # exclude validation time from throughput

                if state.global_step >= total_steps:
                    done = True
                    break

            if not done or state.micro_batches_into_epoch >= micro_batches_per_epoch:
                state.epoch += 1
                state.micro_batches_into_epoch = 0
    except NotImplementedError as error:
        if device.type == "mps":
            raise RuntimeError(
                "An operation is not implemented on MPS (details above). This project deliberately does not enable "
                "PYTORCH_ENABLE_MPS_FALLBACK. Options: run with --device cpu, or set the env var knowingly "
                "(ops will silently run on CPU)."
            ) from error
        raise

    if last_validated_step != state.global_step:
        validate_and_checkpoint()
    training_seconds = time.perf_counter() - training_start

    # Visualize validation predictions of the *selected* (best) checkpoint.
    best = load_checkpoint(checkpoint_dir / "best.pt")
    model.load_state_dict(best["model"])
    best_result = evaluate(model, val_loader, device, cfg.train.amp_dtype)
    val_manifest = data.manifests["val"]
    shown = _gallery_indices(len(best_result.targets), cfg.eval.num_gallery_examples, cfg.eval.gallery_seed)
    gallery_path = viz.plot_sequence_gallery(
        viz.manifest_gallery_rows(val_manifest, data.arrays.images["train"], best_result.indices[shown], best_result.predictions[shown]),
        run_dir / "val_predictions_best.png",
        f"Validation predictions, best checkpoint (step {state.best_step}); random rows, not only successes",
    )
    logger.log_image("examples/val_predictions_best", gallery_path, step=state.global_step)
    curves_path = viz.plot_loss_curves(read_metrics(logger.metrics_path), run_dir / "loss_curves.png", f"{run_dir.name}: loss vs optimizer step")
    logger.log_image("curves/loss", curves_path, step=state.global_step)
    logger.log_checkpoint_artifact(checkpoint_dir / "best.pt", name=f"{run_dir.name}-best")

    summary = {
        "run_dir": str(run_dir),
        "config_name": cfg.name,
        "augment_mode": cfg.augment.mode,
        "seed": cfg.train.seed,
        "device": str(device),
        "parameters": parameter_counts["total"],
        "final_optimizer_step": state.global_step,
        "epochs_completed": state.epoch,
        "best_step": state.best_step,
        "best_val_accuracy": state.best_val_accuracy,
        "best_val_loss": state.best_val_loss,
        "effective_batch_size": cfg.train.batch_size * accumulation,
        "training_seconds_this_session": round(training_seconds, 1),
        "data_ids": data_ids,
        "note": "Validation selected best.pt. Test accuracy requires running evaluate.py --split test explicitly.",
    }
    logger.log_summary(summary)
    logger.finish()
    print(f"Done. best val acc {state.best_val_accuracy:.4f} at step {state.best_step}. Checkpoints: {checkpoint_dir}")
    return run_dir
