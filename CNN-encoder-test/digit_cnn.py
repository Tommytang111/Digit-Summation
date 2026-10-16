"""Supervised digit-classifier reference for the sum model's vision encoder.

Trains the *same* CNN as the sum model's image encoder (model.ImageEncoder plus a 10-way linear
head) with the *same* Albumentations training augmentation. It trains directly on MNIST digit
labels, using the *same* 50,000-image training pool the sum model draws from. The best epoch is
picked on the project's 10,000-image validation pool, and the official 10,000-image test set is
evaluated once.

    ../.torch/bin/python CNN-encoder-test/digit_cnn.py --run-name relu-max-bn-s0     # the config's encoder (now ReLU/max/BN)
    ../.torch/bin/python CNN-encoder-test/digit_cnn.py --activation gelu --pooling avg --no-batch-norm --run-name digit-cnn-s0  # original
    ../.torch/bin/python CNN-encoder-test/digit_cnn.py --epochs 1 --max-steps 30 --device cpu --run-name smoke

Why: the sum model learns digit identity only through the sum loss. This run shows what digit
accuracy and macro F1 this architecture + augmentation reach when trained on digit labels.
It also computes the *implied sum ceiling*: the exact-sum accuracy on the sum model's own val/test
sequences if every image operand were read with this CNN's accuracy and addition were perfect.

Outputs go to CNN-encoder-test/runs/<run-name>/ (metrics.json, best.pt, test_predictions.csv,
curves.png, confusion_test.png, misclassified_test.png).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))  # reuse the project's modules (model, augmentations, engine, ...)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

NUM_DIGITS = 10


# ----------------------------------------------------------------------------- model + data


class DigitCNN(nn.Module):
    """model.ImageEncoder (the sum model's encoder class) + Linear(d_model -> 10).

    images float32 [N, 1, 28, 28] -> encoder -> [N, d_model] -> head -> logits [N, 10]

    The encoder options (activation, pooling, BatchNorm) come from the config's model section,
    the same fields the sum model uses. The CLI flags --activation / --pooling / --[no-]batch-norm
    override them for one run.
    """

    def __init__(self, model_cfg):
        super().__init__()
        from model import ImageEncoder

        self.encoder = ImageEncoder(
            model_cfg.d_model,
            model_cfg.conv1_channels,
            model_cfg.conv2_channels,
            pooling=model_cfg.cnn_pooling,
            activation=model_cfg.cnn_activation,
            batch_norm=model_cfg.cnn_batch_norm,
        )
        self.head = nn.Linear(model_cfg.d_model, NUM_DIGITS)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(images))


class DigitImageDataset(Dataset):
    """Single MNIST images from one pool.

    Keys: an integer position, or a (position, epoch) pair (what engine.EpochShuffleSampler
    yields). With (position, epoch) and an augment_seed, the transform is reseeded with
    derive_seed(augment_seed, epoch, image_id), the same scheme the sum model uses. Augmentation
    therefore does not depend on num_workers.

    Returns (pixels float32 [1, 28, 28], label int, image_id int). image_id indexes the official
    partition the pool belongs to.
    """

    def __init__(self, partition_images: np.ndarray, partition_labels: np.ndarray, pool_ids: np.ndarray, transform, augment_seed=None):
        self.images = partition_images
        self.labels = partition_labels
        self.ids = np.asarray(pool_ids, dtype=np.int64)
        self.transform = transform  # name `transform` is what augmentations.seed_worker looks for
        self.augment_seed = augment_seed

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, key):
        from augmentations import derive_seed

        position, epoch = (int(key[0]), int(key[1])) if isinstance(key, tuple) else (int(key), None)
        image_id = int(self.ids[position])
        if epoch is not None and self.augment_seed is not None:
            self.transform.set_random_seed(derive_seed(self.augment_seed, epoch, image_id))
        pixels = self.transform(self.images[image_id])  # float32 [28, 28]
        return torch.from_numpy(pixels).unsqueeze(0), int(self.labels[image_id]), image_id


def make_loader(dataset, batch_size, train_cfg, device, sampler=None, drop_last=False, seed=0) -> DataLoader:
    from augmentations import seed_worker

    workers = train_cfg.num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=drop_last,
        num_workers=workers,
        persistent_workers=bool(train_cfg.persistent_workers and workers > 0),
        pin_memory=bool(train_cfg.pin_memory and device.type == "cuda"),
        worker_init_fn=seed_worker if workers > 0 else None,
        generator=torch.Generator().manual_seed(seed),
    )


# ----------------------------------------------------------------------------- metrics


def digit_report(predictions: np.ndarray, labels: np.ndarray) -> dict:
    """Accuracy, macro F1, per-digit precision/recall/F1/support, and the 10x10 confusion matrix."""
    from engine import macro_f1

    confusion = np.zeros((NUM_DIGITS, NUM_DIGITS), dtype=np.int64)  # rows = true digit, cols = predicted
    np.add.at(confusion, (labels, predictions), 1)
    true_positives = np.diag(confusion)
    predicted = confusion.sum(axis=0)
    actual = confusion.sum(axis=1)
    per_digit = []
    for digit in range(NUM_DIGITS):
        precision = true_positives[digit] / predicted[digit] if predicted[digit] else 0.0
        recall = true_positives[digit] / actual[digit] if actual[digit] else 0.0
        f1 = 2 * true_positives[digit] / (predicted[digit] + actual[digit]) if predicted[digit] + actual[digit] else 0.0
        per_digit.append(
            {"digit": digit, "precision": float(precision), "recall": float(recall), "f1": float(f1), "support": int(actual[digit])}
        )
    return {
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": macro_f1(predictions, labels, NUM_DIGITS),
        "n": int(len(labels)),
        "errors": int((predictions != labels).sum()),
        "per_digit": per_digit,
        "confusion": confusion.tolist(),
    }


def implied_sum_report(manifest, digit_prediction_by_id: np.ndarray, image_accuracy: float) -> dict:
    """Exact-sum metrics if image operands were read by this CNN and addition were perfect.

    digit_prediction_by_id: int64 [partition size], the CNN's predicted digit per image ID (-1 if
    not predicted). Text operands keep their true value. Also reports the independent-errors
    approximation p^k (p = per-image accuracy, k = images in the row). It is usually a bit below
    the actual value, because two wrong digits can still produce the right sum.
    """
    from engine import macro_f1

    values = manifest.digits.astype(np.int64).copy()
    image_slots = manifest.is_image
    read_digits = digit_prediction_by_id[manifest.source_index[image_slots]]
    if (read_digits < 0).any():
        raise ValueError(f"{manifest.name}: some image operands have no CNN prediction")
    values[image_slots] = read_digits
    predicted_sums = values.sum(axis=1)
    targets = manifest.targets.astype(np.int64)
    correct = predicted_sums == targets
    image_counts = image_slots.sum(axis=1)
    by_images = []
    for k in range(manifest.sequence_length + 1):
        rows = image_counts == k
        if rows.any():
            by_images.append({"images": k, "n": int(rows.sum()), "accuracy": float(correct[rows].mean()), "approx_p_pow_k": image_accuracy**k})
    return {
        "manifest": manifest.name,
        "n": int(len(targets)),
        "exact_sum_accuracy": float(correct.mean()),
        "sum_macro_f1": macro_f1(predicted_sums, targets, 9 * manifest.sequence_length + 1),
        "approx_independent_errors": float(np.mean(image_accuracy**image_counts)),
        "by_image_count": by_images,
    }


@torch.no_grad()
def predict(model, loader, device):
    """Eval-mode pass: mean loss, predictions, labels, image IDs, and max softmax probability."""
    model.eval()
    loss_sum, outputs = 0.0, {"pred": [], "label": [], "id": [], "conf": []}
    for pixels, labels, ids in loader:
        logits = model(pixels.to(device)).float()
        labels_device = labels.to(device)
        loss_sum += F.cross_entropy(logits, labels_device, reduction="sum").item()
        probabilities = logits.softmax(dim=-1)
        outputs["pred"].append(logits.argmax(dim=-1).cpu())
        outputs["conf"].append(probabilities.max(dim=-1).values.cpu())
        outputs["label"].append(labels)
        outputs["id"].append(ids)
    merged = {k: torch.cat(v).numpy() for k, v in outputs.items()}
    return loss_sum / len(merged["label"]), merged


# ----------------------------------------------------------------------------- figures


def plot_curves(history: list[dict], path: Path) -> None:
    import visualization as viz

    fig, axes = viz.plt.subplots(1, 3, figsize=(16, 3.8))
    for ax, key, label in zip(axes, ("loss", "accuracy", "macro_f1"), ("Cross-entropy loss", "Digit accuracy", "Digit macro F1")):
        steps = [h["optimizer_step"] for h in history]
        ax.plot(steps, [h[f"train_{key}"] for h in history], color=viz.SERIES["train"], label="train (epoch, augmented)")
        ax.plot(steps, [h[f"val_{key}"] for h in history], color=viz.SERIES["val"], marker="o", markersize=4, label="validation")
        ax.set_xlabel("Optimizer step (parameter updates)")
        ax.set_ylabel(label)
        viz._style(ax)
        ax.legend(loc="best")
    fig.suptitle("Digit CNN (sum-model encoder architecture): training curves", color=viz.INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    viz._save(fig, path)


def plot_confusion(confusion: np.ndarray, path: Path, title: str) -> None:
    """Off-diagonal error counts on a single-hue blue ramp; the diagonal shows correct counts uncolored."""
    import visualization as viz
    from matplotlib.colors import LinearSegmentedColormap

    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # blue 100 -> 700
    cmap = LinearSegmentedColormap.from_list("blue_ramp", ramp)
    errors = confusion.astype(float).copy()
    np.fill_diagonal(errors, np.nan)
    vmax = max(1.0, np.nanmax(errors))
    fig, ax = viz.plt.subplots(figsize=(6.4, 5.6))
    shown = np.ma.masked_where(np.isnan(errors) | (errors == 0), errors)
    image = ax.imshow(shown, cmap=cmap, vmin=0.5, vmax=vmax)
    for t in range(NUM_DIGITS):
        for p in range(NUM_DIGITS):
            value = int(confusion[t, p])
            if t == p:
                ax.text(p, t, f"{value}", ha="center", va="center", fontsize=7, color=viz.INK_SECONDARY, fontweight="bold")
            elif value:
                dark = value > 0.55 * vmax
                ax.text(p, t, f"{value}", ha="center", va="center", fontsize=8, color="#ffffff" if dark else viz.INK)
    ax.set_xticks(range(NUM_DIGITS))
    ax.set_yticks(range(NUM_DIGITS))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title(title, loc="left", fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Misclassified images")
    fig.tight_layout()
    viz._save(fig, path)


def plot_misclassified(images: np.ndarray, rows: list[dict], path: Path, title: str) -> None:
    import visualization as viz

    columns = 8
    count = max(1, len(rows))
    fig, axes = viz.plt.subplots((count + columns - 1) // columns, columns, figsize=(1.35 * columns, 1.6 * ((count + columns - 1) // columns) + 0.5), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, row in zip(axes.ravel(), rows):
        ax.imshow(images[row["image_id"]], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"true {row['label']} → {row['prediction']} ✗\np={row['confidence']:.2f}", fontsize=7.5, color=viz.CRITICAL)
    if not rows:
        axes[0, 0].text(0.5, 0.5, "no test errors", ha="center", va="center", transform=axes[0, 0].transAxes)
    fig.suptitle(title, color=viz.INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    viz._save(fig, path)


# ----------------------------------------------------------------------------- main


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    from augmentations import build_transform, derive_seed
    from config import config_to_dict, load_config
    from dataset import load_manifests, load_mnist_arrays, load_source_pools
    from engine import EpochShuffleSampler, build_optimizer, build_scheduler, resolve_device, seed_everything, synchronize, warn_if_mps_fallback

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "local.yaml"), help="source of architecture, augmentation, data, optimizer settings")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=None, help="cap on optimizer steps (smoke tests)")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--gallery-size", type=int, default=32)
    parser.add_argument("--activation", default=None, choices=["relu", "gelu"], help="overrides model.cnn_activation")
    parser.add_argument("--pooling", default=None, choices=["avg", "max"], help="overrides model.cnn_pooling")
    parser.add_argument("--batch-norm", default=None, action=argparse.BooleanOptionalAction, help="overrides model.cnn_batch_norm")
    args = parser.parse_args()

    for flag, key in ((args.activation, "cnn_activation"), (args.pooling, "cnn_pooling")):
        if flag is not None:
            args.overrides = [*args.overrides, f"model.{key}={flag}"]
    if args.batch_norm is not None:
        args.overrides = [*args.overrides, f"model.cnn_batch_norm={str(args.batch_norm).lower()}"]
    cfg = load_config(args.config, args.overrides)
    d = cfg.data
    if (d.train_source_fraction, d.val_source_fraction, d.test_source_fraction) != (1.0, 1.0, 1.0):
        raise ValueError("This reference uses the full pools; set all data.*_source_fraction to 1.0")
    device = resolve_device(args.device)
    warn_if_mps_fallback(device)
    seed_everything(cfg.train.seed)

    manifest_dir = resolve(d.manifest_dir)
    arrays = load_mnist_arrays(resolve(d.root))
    pools = load_source_pools(manifest_dir)
    manifests = load_manifests(manifest_dir)
    pool_sizes = {split: int(len(pools.ids[split])) for split in ("train", "val", "test")}
    print(f"Pools from {manifest_dir} (pool {pools.pool_id()}): {pool_sizes}")

    augment_seed = derive_seed(cfg.train.seed, 1)
    train_set = DigitImageDataset(arrays.images["train"], arrays.labels["train"], pools.ids["train"], build_transform(cfg.augment, train=True), augment_seed)
    eval_transform = build_transform(cfg.augment, train=False)
    val_set = DigitImageDataset(arrays.images["train"], arrays.labels["train"], pools.ids["val"], eval_transform)
    test_set = DigitImageDataset(arrays.images["test"], arrays.labels["test"], pools.ids["test"], eval_transform)

    sampler = EpochShuffleSampler(len(train_set), seed=derive_seed(cfg.train.seed, 2))
    train_loader = make_loader(train_set, args.batch_size, cfg.train, device, sampler=sampler, drop_last=True)
    val_loader = make_loader(val_set, 512, cfg.train, device)

    torch.manual_seed(cfg.train.seed)
    model = DigitCNN(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    steps_per_epoch = len(train_set) // args.batch_size
    total_steps = args.epochs * steps_per_epoch if args.max_steps is None else min(args.epochs * steps_per_epoch, args.max_steps)
    scheduler = build_scheduler(optimizer, cfg.train, total_steps)

    run_dir = HERE / "runs" / (args.run_name or f"digit-cnn-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; choose another --run-name")
    run_dir.mkdir(parents=True)
    print(
        f"Device {device} | DigitCNN params {sum(p.numel() for p in model.parameters()):,} "
        f"(encoder {sum(p.numel() for p in model.encoder.parameters()):,}) | encoder activation={cfg.model.cnn_activation}, pooling={cfg.model.cnn_pooling}, "
        f"batch_norm={cfg.model.cnn_batch_norm} | augment={cfg.augment.mode}\n"
        f"{steps_per_epoch} steps/epoch x {args.epochs} epochs -> {total_steps} optimizer steps, batch {args.batch_size}, "
        f"lr {cfg.train.lr} ({cfg.train.lr_schedule}, warmup {cfg.train.warmup_steps}), workers {cfg.train.num_workers}"
    )

    history, best, step, start = [], {"val_accuracy": -1.0, "val_loss": float("inf")}, 0, time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        loss_sum, predictions, labels_seen = torch.zeros((), device=device), [], []
        for pixels, labels, _ in train_loader:
            pixels, labels = pixels.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(pixels)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.train.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            loss_sum += loss.detach() * labels.numel()
            predictions.append(logits.argmax(dim=-1).detach())
            labels_seen.append(labels)
            if step >= total_steps:
                break
        synchronize(device)
        train_pred, train_true = torch.cat(predictions).cpu().numpy(), torch.cat(labels_seen).cpu().numpy()
        train_metrics = digit_report(train_pred, train_true)
        val_loss, val_out = predict(model, val_loader, device)
        val_metrics = digit_report(val_out["pred"], val_out["label"])
        record = {
            "epoch": epoch + 1,
            "optimizer_step": step,
            "train_loss": float(loss_sum.item()) / len(train_true),
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "elapsed_s": round(time.perf_counter() - start, 1),
        }
        history.append(record)
        improved = val_metrics["accuracy"] > best["val_accuracy"] or (
            val_metrics["accuracy"] == best["val_accuracy"] and val_loss < best["val_loss"]
        )
        if improved:
            best = {"epoch": epoch + 1, "optimizer_step": step, "val_accuracy": val_metrics["accuracy"], "val_loss": val_loss}
            torch.save(
                {"model": model.state_dict(), "best": best, "config": config_to_dict(cfg), "args": vars(args), "pool_id": pools.pool_id()},
                run_dir / "best.pt",
            )
        print(
            f"epoch {epoch + 1:>3} step {step:>6}  train loss {record['train_loss']:.4f} acc {record['train_accuracy']:.4f}  |  "
            f"val loss {val_loss:.4f} acc {record['val_accuracy']:.4f} macro-F1 {record['val_macro_f1']:.4f}"
            f"{'  *best*' if improved else ''}  ({record['elapsed_s']} s)"
        )
        if step >= total_steps:
            break

    # Test exactly once, with the validation-selected weights.
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)["model"])
    val_loss, val_out = predict(model, val_loader, device)
    test_loss, test_out = predict(model, make_loader(test_set, 512, cfg.train, device), device)
    val_report = {**digit_report(val_out["pred"], val_out["label"]), "loss": val_loss}
    test_report = {**digit_report(test_out["pred"], test_out["label"]), "loss": test_loss}

    # Implied sum ceilings on the sum model's own sequences.
    ceilings = {}
    for split, out, partition in (("val", val_out, "train"), ("test", test_out, "test")):
        by_id = np.full(len(arrays.labels[partition]), -1, dtype=np.int64)
        by_id[out["id"]] = out["pred"]
        accuracy = float((out["pred"] == out["label"]).mean())
        for name in (split, f"{split}_all_image"):
            ceilings[name] = implied_sum_report(manifests[name], by_id, accuracy)

    metrics = {
        "what": "Supervised digit CNN with the sum model's ImageEncoder architecture and training augmentation",
        "selection": "best epoch by validation accuracy (ties: lower loss); test evaluated once with those weights",
        "best": best,
        "val": val_report,
        "test": test_report,
        "implied_sum_ceilings": ceilings,
        "history": history,
        "pool_id": pools.pool_id(),
        "pool_sizes": pool_sizes,
        "manifest_dir": str(d.manifest_dir),
        "settings": {**vars(args), "encoder": {"activation": cfg.model.cnn_activation, "pooling": cfg.model.cnn_pooling, "batch_norm": cfg.model.cnn_batch_norm}, "augment": config_to_dict(cfg)["augment"], "model": config_to_dict(cfg)["model"], "device": str(device)},
        "note": "Single seed; seed-to-seed variation not measured.",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    with open(run_dir / "test_predictions.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test_image_id", "label", "prediction", "correct", "confidence"])
        for image_id, label, prediction, confidence in zip(test_out["id"], test_out["label"], test_out["pred"], test_out["conf"]):
            writer.writerow([int(image_id), int(label), int(prediction), int(label == prediction), f"{confidence:.4f}"])

    plot_curves(history, run_dir / "curves.png")
    plot_confusion(np.array(test_report["confusion"]), run_dir / "confusion_test.png",
                   f"Test confusion: {test_report['errors']} errors / {test_report['n']:,} (diagonal = correct, uncolored)")
    errors = np.flatnonzero(test_out["pred"] != test_out["label"])
    shown = np.sort(np.random.default_rng(0).choice(errors, size=min(args.gallery_size, len(errors)), replace=False)) if len(errors) else []
    plot_misclassified(
        arrays.images["test"],
        [{"image_id": int(test_out["id"][i]), "label": int(test_out["label"][i]), "prediction": int(test_out["pred"][i]), "confidence": float(test_out["conf"][i])} for i in shown],
        run_dir / "misclassified_test.png",
        f"Test misclassifications: {len(shown)} random of {len(errors)} (raw pixels; p = model confidence)",
    )

    print(f"\n== Best epoch {best['epoch']} (step {best['optimizer_step']}), selected on validation ==")
    print(f"val   digit accuracy {val_report['accuracy']:.4f}  macro F1 {val_report['macro_f1']:.4f}  ({val_report['errors']} errors / {val_report['n']:,})")
    print(f"TEST  digit accuracy {test_report['accuracy']:.4f}  macro F1 {test_report['macro_f1']:.4f}  ({test_report['errors']} errors / {test_report['n']:,})")
    print("test per-digit F1: " + "  ".join(f"{r['digit']}:{r['f1']:.4f}" for r in test_report["per_digit"]))
    print("\n== Implied sum ceilings (CNN reads every image, addition perfect) ==")
    for name, c in ceilings.items():
        print(f"{name:>15}: exact-sum accuracy {c['exact_sum_accuracy']:.4f}  sum macro F1 {c['sum_macro_f1']:.4f}  (p^k approx {c['approx_independent_errors']:.4f}, n={c['n']:,})")
    print(f"\nWrote {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
