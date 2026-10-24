"""Typed experiment configuration: YAML file -> nested dataclasses -> validation.

Usage:
    cfg = load_config("configs/debug.yaml", overrides=["train.lr=1e-3", "augment.mode=none"])

Design notes:
- Every YAML key must correspond to a dataclass field; unknown keys raise immediately, so a typo
  such as `train.learning_rate` cannot be silently ignored.
- Values are type-checked against the field annotations. Numeric strings are accepted for float
  fields because PyYAML (YAML 1.1) parses `3e-4` as the *string* "3e-4" (it requires `3.0e-4`).
- `--set key.path=value` overrides are parsed with yaml.safe_load, so `true`, `null`, `64`, and
  `[1, 2]` behave as they would in the YAML file.
"""

from __future__ import annotations

import dataclasses
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Union, get_args, get_origin, get_type_hints

import yaml

AugmentMode = Literal["none", "affine", "affine_elastic"]
DeviceName = Literal["auto", "mps", "cpu", "cuda"]
WandbMode = Literal["disabled", "offline", "online"]


class ConfigError(ValueError):
    """Raised for unknown keys, wrong types, or invalid value combinations."""


@dataclass
class DataConfig:
    """Source-image pools and generated arithmetic-sequence manifests."""

    root: str = "data"  # torchvision downloads MNIST to <root>/MNIST/raw
    manifest_dir: str = "data/manifests/debug"  # source pools + manifests + audit live here
    sequence_length: int = 10  # L: operands per example (main task: exactly 10)
    image_probability: float = 0.5  # P(operand is an MNIST image) independently per position
    val_source_size: int = 10_000  # official-train images held out as the validation pool
    # Stratified fraction of each source pool that sequences may draw from (nested subsets).
    train_source_fraction: float = 1.0
    val_source_fraction: float = 1.0
    test_source_fraction: float = 1.0
    num_train_sequences: int = 20_000
    num_val_sequences: int = 2_000
    num_test_sequences: int = 5_000
    num_diagnostic_sequences: int = 1_000  # size of each text-only / all-image diagnostic set
    # Distinct, recorded seeds. split_seed controls the 50k/10k split and fraction subsets.
    split_seed: int = 0
    train_seed: int = 1_001
    val_seed: int = 2_002
    test_seed: int = 3_003
    diagnostic_seed: int = 4_004


@dataclass
class AugmentConfig:
    """Albumentations settings. Applied to *training* images only; val/test only normalize."""

    mode: AugmentMode = "affine"
    rotate_degrees: float = 10.0  # rotation sampled uniformly in [-r, r]
    translate_px: int = 2  # integer pixel shift sampled in [-t, t] per axis
    scale_min: float = 0.9  # per-axis scale (independent x/y => mild squeeze)
    scale_max: float = 1.1
    shear_degrees: float = 5.0  # per-axis shear in [-s, s]
    affine_p: float = 1.0
    # Elastic settings are in albumentations 2.0.8 units (see augmentations.py and README):
    # displacement = alpha * GaussianBlur(uniform(-1, 1) noise, sigma), in pixels.
    elastic_alpha: float = 12.0
    elastic_sigma: float = 4.0
    elastic_p: float = 0.5


@dataclass
class ModelConfig:
    """Small CNN + Transformer encoder trained from scratch."""

    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    conv1_channels: int = 32
    conv2_channels: int = 64
    cnn_pooling: Literal["avg", "max"] = "max"  # 2x2 downsampling after each conv
    cnn_activation: Literal["relu", "gelu"] = "relu"
    cnn_batch_norm: bool = True  # BatchNorm after each conv (conv bias dropped)
    use_modality_embedding: bool = True
    norm_first: bool = True  # pre-LayerNorm Transformer blocks (+ final LayerNorm)


@dataclass
class TrainConfig:
    """Optimization, device, and schedule settings shared by local and cloud entry points."""

    device: DeviceName = "auto"
    seed: int = 0  # model init, shuffling, augmentation, dropout
    batch_size: int = 64  # micro-batch size
    grad_accum_steps: int = 1  # effective batch = batch_size * grad_accum_steps
    epochs: int = 1
    max_steps: Optional[int] = None  # cap on optimizer updates (None = run all epochs)
    eval_every_steps: int = 500  # validation frequency, in optimizer updates
    log_every_steps: int = 50
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip_norm: Optional[float] = 1.0
    lr_schedule: Literal["constant", "warmup_cosine"] = "constant"
    warmup_steps: int = 0
    # Early stopping: stop after this many consecutive validations without a new highest
    # validation accuracy (higher by more than min_delta, in accuracy units, e.g. 0.001 = 0.1
    # points). null disables it. The count is in validations, so patience 15 with
    # eval_every_steps 1000 means 15,000 updates without improvement.
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0
    num_workers: int = 0
    # Keep DataLoader worker processes alive across epochs/validations (num_workers > 0 only).
    # Affects speed only: augmentation is seeded per (epoch, row), not per worker.
    persistent_workers: bool = False
    pin_memory: bool = False  # only honored on CUDA
    amp_dtype: Literal["none", "bf16"] = "none"  # bf16 autocast is CUDA-only
    allow_tf32: bool = False  # CUDA-only; changes float32 matmul numerics


@dataclass
class LoggingConfig:
    """Local records are always written; W&B is optional."""

    output_dir: str = "runs"
    run_name: Optional[str] = None  # default: <config name>-<timestamp>
    wandb_mode: WandbMode = "disabled"
    wandb_project: str = "digit-summation"
    wandb_entity: Optional[str] = None
    upload_checkpoints: bool = False  # log best.pt as a W&B artifact (online/offline modes)


@dataclass
class OverfitConfig:
    """Tiny-overfit wiring check (sanity_checks.py overfit): aug and dropout are forced off."""

    num_examples: int = 64
    max_steps: int = 300
    lr: float = 1e-3
    batch_size: int = 64
    eval_every_steps: int = 10
    target_accuracy: float = 1.0
    seed: int = 5
    # null: sample rows of the training manifest. A number p: generate fresh rows from the
    # *training pool* with image probability p (1.0 isolates the image path).
    image_probability: Optional[float] = None


@dataclass
class EvalConfig:
    """evaluate.py settings."""

    batch_size: int = 256
    num_gallery_examples: int = 16
    gallery_seed: int = 7
    permutation_seed: int = 11
    stress_seed: int = 13
    num_stress_image_sequences: int = 50


BonusArchitecture = Literal[
    "items_only_sinusoidal", "items_only_no_positions", "sinusoidal_plus_sum_summary",
    "items_only_rope", "items_only_alibi",
]


@dataclass
class BonusConfig:
    """Length-generalization bonus: variable-length operand sequences -> sum written by a decoder.

    Only the bonus entry points read this section; the main-task scripts ignore it. The source
    pools, image probability, seeds, and augmentation come from the data/augment sections, and the
    operand encoder (CNN, embeddings, encoder layers) from the model section.
    """

    architecture: BonusArchitecture = "items_only_sinusoidal"
    manifest_dir: str = "data/manifests/bonus"
    min_train_length: int = 1
    max_train_length: int = 10  # training lengths are uniform over [min, max] by default
    max_eval_length: int = 20  # evaluation at every length 1..max_eval_length
    num_train_sequences: int = 100_000  # total, split evenly across training lengths
    num_eval_per_length: int = 1_000  # val and test, each length
    num_all_image_per_length: int = 200  # val and test all-image diagnostic sets, each length
    decoder_layers: int = 2
    decoder_heads: int = 4
    decoder_feedforward: int = 512
    decoder_dropout: float = 0.1
    max_new_tokens: int = 4  # generation cap after BOS: three digits + EOS (sums up to 180)
    summary_scale: float = 0.1  # sum-summary ablation only: fixed scale, identical at all lengths
    eval_batch_size: int = 512


@dataclass
class ExperimentConfig:
    name: str = "debug"
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    overfit: OverfitConfig = field(default_factory=OverfitConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    bonus: BonusConfig = field(default_factory=BonusConfig)

    @property
    def num_classes(self) -> int:
        """Sums range over 0 .. 9*L, so 91 classes for L=10."""
        return 9 * self.data.sequence_length + 1


# ----------------------------------------------------------------------------- loading


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    """Check (and lightly coerce) one YAML value against a dataclass field annotation."""
    origin = get_origin(annotation)

    if origin in (Union, types.UnionType):  # Optional[X] == Union[X, None]
        args = [a for a in get_args(annotation) if a is not type(None)]
        if value is None:
            return None
        return _coerce(value, args[0], path)

    if origin is Literal:
        allowed = get_args(annotation)
        if value not in allowed:
            raise ConfigError(f"{path}: {value!r} is not one of {allowed}")
        return value

    if dataclasses.is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        return _build(annotation, value, path)

    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if annotation is float:
        if isinstance(value, bool):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)  # handles YAML-1.1 strings such as "3e-4"
            except ValueError:
                pass
        raise ConfigError(f"{path}: expected a number, got {value!r}")
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{path}: unsupported annotation {annotation!r}")


def _build(cls: type, raw: dict, path: str) -> Any:
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"{path or 'config'}: unknown key(s) {unknown}; valid keys: {sorted(known)}")
    kwargs = {key: _coerce(value, hints[key], f"{path}.{key}" if path else key) for key, value in raw.items()}
    return cls(**kwargs)


def apply_override(raw: dict, override: str) -> None:
    """Apply one `dotted.key=value` override in place to a raw (YAML-loaded) config dict."""
    if "=" not in override:
        raise ConfigError(f"Override {override!r} must look like section.key=value")
    key_path, value_text = override.split("=", 1)
    keys = key_path.strip().split(".")
    node = raw
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ConfigError(f"Override {override!r}: {key!r} is not a section")
    node[keys[-1]] = yaml.safe_load(value_text)


def config_from_dict(raw: dict) -> ExperimentConfig:
    """Build and validate a config from a plain dict (e.g. one stored in a checkpoint)."""
    cfg = _build(ExperimentConfig, raw, "")
    validate(cfg)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load a YAML config, apply `section.key=value` overrides, then validate."""
    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}
    for override in overrides or []:
        apply_override(raw, override)
    return config_from_dict(raw)


def config_to_dict(cfg: ExperimentConfig) -> dict:
    """Plain-dict view (JSON/YAML/W&B friendly). Contains no credentials by construction."""
    return dataclasses.asdict(cfg)


# ----------------------------------------------------------------------------- validation


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def validate(cfg: ExperimentConfig) -> None:
    """Reject invalid values and combinations early, with readable messages."""
    d, a, m, t, o = cfg.data, cfg.augment, cfg.model, cfg.train, cfg.overfit

    _require(d.sequence_length >= 1, "data.sequence_length must be >= 1")
    _require(0.0 <= d.image_probability <= 1.0, "data.image_probability must be in [0, 1]")
    _require(d.val_source_size >= 10, "data.val_source_size must be >= 10 (one per digit)")
    for name in ("train_source_fraction", "val_source_fraction", "test_source_fraction"):
        value = getattr(d, name)
        _require(0.0 < value <= 1.0, f"data.{name} must be in (0, 1], got {value}")
    for name in ("num_train_sequences", "num_val_sequences", "num_test_sequences", "num_diagnostic_sequences"):
        _require(getattr(d, name) >= 1, f"data.{name} must be >= 1")
    seeds = [d.train_seed, d.val_seed, d.test_seed, d.diagnostic_seed]
    _require(len(set(seeds)) == len(seeds), "data.train/val/test/diagnostic seeds must be distinct")

    _require(a.rotate_degrees >= 0 and a.translate_px >= 0 and a.shear_degrees >= 0, "augment ranges must be >= 0")
    _require(0.0 < a.scale_min <= a.scale_max, "augment: need 0 < scale_min <= scale_max")
    _require(0.0 <= a.affine_p <= 1.0 and 0.0 <= a.elastic_p <= 1.0, "augment probabilities must be in [0, 1]")
    _require(a.elastic_alpha >= 0.0, "augment.elastic_alpha must be >= 0")
    _require(a.elastic_sigma >= 1.0, "augment.elastic_sigma must be >= 1 (albumentations constraint)")

    _require(m.nhead >= 1 and m.d_model >= 2 and m.d_model % m.nhead == 0, "model.d_model must be >= 2 and divisible by positive model.nhead")
    _require(m.num_layers >= 1, "model.num_layers must be >= 1")
    _require(0.0 <= m.dropout < 1.0, "model.dropout must be in [0, 1)")

    _require(t.batch_size >= 1 and t.grad_accum_steps >= 1 and t.epochs >= 1, "train sizes must be >= 1")
    _require(t.max_steps is None or t.max_steps >= 1, "train.max_steps must be >= 1 or null")
    _require(t.eval_every_steps >= 1 and t.log_every_steps >= 1, "train eval/log frequencies must be >= 1")
    _require(t.lr > 0 and t.weight_decay >= 0, "train.lr must be > 0 and weight_decay >= 0")
    _require(t.grad_clip_norm is None or t.grad_clip_norm > 0, "train.grad_clip_norm must be > 0 or null")
    _require(t.warmup_steps >= 0 and t.num_workers >= 0, "train.warmup_steps/num_workers must be >= 0")
    _require(t.early_stopping_patience is None or t.early_stopping_patience >= 1, "train.early_stopping_patience must be >= 1 or null")
    _require(0 <= t.early_stopping_min_delta < 1, "train.early_stopping_min_delta (accuracy units) must be in [0, 1)")
    if t.amp_dtype != "none" or t.allow_tf32:
        _require(t.device in ("cuda", "auto"), "bf16 autocast / TF32 are CUDA-only; set train.device=cuda")

    _require(o.num_examples >= 1 and o.max_steps >= 1 and o.batch_size >= 1, "overfit sizes must be >= 1")
    _require(0.0 < o.target_accuracy <= 1.0, "overfit.target_accuracy must be in (0, 1]")
    _require(o.image_probability is None or 0.0 <= o.image_probability <= 1.0, "overfit.image_probability must be in [0, 1] or null")

    b = cfg.bonus
    _require(1 <= b.min_train_length <= b.max_train_length <= b.max_eval_length, "bonus lengths need 1 <= min_train <= max_train <= max_eval")
    _require(b.num_train_sequences >= b.max_train_length - b.min_train_length + 1, "bonus.num_train_sequences must give >= 1 row per training length")
    _require(b.num_eval_per_length >= 1 and b.num_all_image_per_length >= 1, "bonus eval sizes must be >= 1")
    _require(b.decoder_layers >= 1 and b.decoder_heads >= 1 and m.d_model % b.decoder_heads == 0, "bonus decoder: >= 1 layer and d_model divisible by positive decoder_heads")
    if b.architecture == "items_only_rope":
        _require((m.d_model // m.nhead) % 2 == 0 and (m.d_model // b.decoder_heads) % 2 == 0,
                 "RoPE requires even encoder and decoder head dimensions")
    _require(0.0 <= b.decoder_dropout < 1.0, "bonus.decoder_dropout must be in [0, 1)")
    # 9 * max_eval_length can need three digits (e.g. 180), so the cap must allow 3 digits + EOS.
    _require(b.max_new_tokens >= len(str(9 * b.max_eval_length)) + 1, "bonus.max_new_tokens must allow the longest possible answer plus EOS")
    _require(b.summary_scale > 0, "bonus.summary_scale must be > 0")
