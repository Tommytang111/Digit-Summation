"""Local machine-readable run records, plus optional Weights & Biases logging.

Always written (no network or credentials needed):
    <run_dir>/config.json     resolved config + seeds + data IDs
    <run_dir>/metrics.jsonl   one JSON object per log event; each carries "optimizer_step"
    <run_dir>/summary.json    final summary (best step, best val accuracy, ...)

W&B modes (logging.wandb_mode):
    disabled  wandb is never imported
    offline   logs to <run_dir>/wandb/; upload later with `wandb sync <run_dir>/wandb/offline-run-*`
    online    streams to wandb.ai using your existing `wandb login` or the WANDB_API_KEY env var

A training run is a W&B run grouped under its run-directory name. Each evaluate.py call is a
separate W&B run (job_type "eval") in the same group, so a run's evaluations sit next to it.

This module never reads, prints, or logs an API key. wandb itself finds the credentials.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

SECRET_MARKERS = ("api_key", "apikey", "token", "secret", "password")


def sanitize(value):
    """Recursively drop dict keys that look like credentials (defensive; configs hold none)."""
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items() if not any(m in str(k).lower() for m in SECRET_MARKERS)}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return value


def read_metrics(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten_metrics(metrics: dict, prefix: str = "") -> dict[str, float]:
    """Nested metrics -> {"prefix/a/b": number}, the flat scalar form W&B summaries need.

    Lists of group rows (e.g. by_image_count: [{"group": 5, "accuracy": ..., "n": ...}]) are keyed
    by their "group". Strings, None values, and other lists carry no scalar and are skipped.
    """
    flat = {}
    for key, value in metrics.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, name))
        elif isinstance(value, list) and value and all(isinstance(item, dict) and "group" in item for item in value):
            for item in value:
                flat.update(flatten_metrics({k: v for k, v in item.items() if k != "group"}, f"{name}/{item['group']}"))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[name] = value
    return flat


def log_evaluation(logging_cfg, run_config: dict, summary: dict, images: dict[str, Path], name: str, group: str, directory: Path) -> None:
    """Log one evaluation as its own W&B run (job_type "eval") in its training run's group.

    `summary` (flat scalars, see flatten_metrics) goes to the run summary, so each metric is one
    value in the W&B runs table rather than a one-point chart. `images` maps W&B keys to PNG
    files. Does nothing when logging_cfg.wandb_mode is "disabled".
    """
    if logging_cfg.wandb_mode == "disabled":
        return
    import wandb  # imported lazily so the disabled mode never needs it

    run = wandb.init(
        project=logging_cfg.wandb_project,
        entity=logging_cfg.wandb_entity,
        name=name,
        group=group,
        job_type="eval",
        mode=logging_cfg.wandb_mode,
        dir=str(directory),
        config=sanitize(run_config),
    )
    run.summary.update(summary)
    run.log({key: wandb.Image(str(path)) for key, path in images.items()})
    run.finish()


class ExperimentLogger:
    """Writes local records and mirrors scalars/images to W&B when enabled.

    The x-axis everywhere is `optimizer_step`: the number of optimizer.step() calls so far.
    Epochs are logged as a separate (fractional) field.
    """

    def __init__(self, run_dir: Path, run_config: dict, logging_cfg, job_type: str = "train", resume: bool = False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.logging_cfg = logging_cfg
        self.start_time = time.time()
        with open(self.run_dir / ("config.json" if not resume else "config_resumed.json"), "w") as handle:
            json.dump(sanitize(run_config), handle, indent=2, default=str)

        self.wandb = None
        self.wandb_run = None
        if logging_cfg.wandb_mode != "disabled":
            import wandb  # imported lazily so the disabled mode never needs it

            self.wandb = wandb
            self.wandb_run = wandb.init(
                project=logging_cfg.wandb_project,
                entity=logging_cfg.wandb_entity,
                name=self.run_dir.name,
                group=self.run_dir.name,  # evaluate.py logs this run's evaluations to the same group
                mode=logging_cfg.wandb_mode,
                dir=str(self.run_dir),
                config=sanitize(run_config),
                job_type=job_type,
            )
            wandb.define_metric("optimizer_step")
            wandb.define_metric("*", step_metric="optimizer_step")

    def truncate_after(self, step: int) -> None:
        """On resume, drop local records logged after the checkpoint's step."""
        kept = [r for r in read_metrics(self.metrics_path) if r.get("optimizer_step", 0) <= step]
        with open(self.metrics_path, "w") as handle:
            for record in kept:
                handle.write(json.dumps(record) + "\n")

    def log_metrics(self, metrics: dict, step: int) -> None:
        record = {"optimizer_step": int(step), "wall_time_s": round(time.time() - self.start_time, 3), **metrics}
        with open(self.metrics_path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        if self.wandb_run is not None:
            self.wandb.log(record)

    def log_image(self, key: str, image_path: Path, step: int) -> None:
        if self.wandb_run is not None:
            self.wandb.log({key: self.wandb.Image(str(image_path)), "optimizer_step": int(step)})

    def log_summary(self, summary: dict) -> None:
        with open(self.run_dir / "summary.json", "w") as handle:
            json.dump(summary, handle, indent=2, default=str)
        if self.wandb_run is not None:
            self.wandb_run.summary.update({k: v for k, v in summary.items() if isinstance(v, (int, float, str, bool))})

    def log_checkpoint_artifact(self, checkpoint_path: Path, name: str) -> None:
        """Optional (logging.upload_checkpoints): store a checkpoint as a W&B model artifact."""
        if self.wandb_run is None or not self.logging_cfg.upload_checkpoints:
            return
        artifact = self.wandb.Artifact(name=name, type="model")
        artifact.add_file(str(checkpoint_path))
        self.wandb_run.log_artifact(artifact)

    def finish(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()
