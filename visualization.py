"""Matplotlib figures (static PNGs) shared by inspection, training, evaluation, and sanity checks.

Style: light chart surface, recessive hairline grid, 2px lines, fixed series colors per split
(train=blue, val=orange, test=aqua; validated as a CVD-safe set). Legends are shown for every
multi-series chart, and correctness is marked with a ✓/✗ glyph, not by color alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed (works headless on servers)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "font.family": "sans-serif",
        "lines.linewidth": 2.0,
    }
)


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- training curves


def plot_loss_curves(records: list[dict], path: Path, title: str = "Training progress") -> Path:
    """Loss (left) and exact-sum accuracy (right) against optimizer steps (parameter updates).

    Training values are means over each logging interval, computed in train mode (dropout and
    augmentation on). Validation values are full passes in eval mode.
    """
    train = [r for r in records if "train/loss" in r]
    val = [r for r in records if "val/loss" in r]
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, label in ((loss_ax, "loss", "Cross-entropy loss"), (acc_ax, "accuracy", "Exact-sum accuracy")):
        if train:
            ax.plot([r["optimizer_step"] for r in train], [r[f"train/{key}"] for r in train], color=SERIES["train"], label="train (interval mean)")
        if val:
            ax.plot(
                [r["optimizer_step"] for r in val],
                [r[f"val/{key}"] for r in val],
                color=SERIES["val"],
                marker="o",
                markersize=4,
                label="validation",
            )
        ax.set_xlabel("Optimizer step (parameter updates)")
        ax.set_ylabel(label)
        _style(ax)
        ax.legend(loc="best")
    acc_ax.set_ylim(-0.02, 1.02)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, path)


def plot_overfit_curve(history: list[dict], path: Path, title: str) -> Path:
    """Tiny-overfit check: loss and accuracy on the fixed examples (eval mode) vs optimizer step."""
    steps = [h["optimizer_step"] for h in history]
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(10, 3.4))
    loss_ax.plot(steps, [h["loss"] for h in history], color=SERIES["train"])
    loss_ax.set_yscale("log")
    loss_ax.set_ylabel("Loss on fixed set (log scale)")
    acc_ax.plot(steps, [h["accuracy"] for h in history], color=SERIES["train"])
    acc_ax.set_ylim(-0.02, 1.02)
    acc_ax.set_ylabel("Accuracy on fixed set")
    for ax in (loss_ax, acc_ax):
        ax.set_xlabel("Optimizer step")
        _style(ax)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------------------------- sequence galleries


def plot_sequence_gallery(rows: list[dict], path: Path, title: str, pixel_range: tuple[float, float] = (0, 255)) -> Path:
    """One row per sequence: image operands as pixels, text operands as their digit string.

    Each row dict has: images (list of [28, 28] arrays or None for text), tokens (list of str for
    text slots or None), target (int), and optionally prediction (int) and caption (str).
    `pixel_range` is (0, 255) for raw pixels or the normalized black/white values for model views.
    """
    if not rows:
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.axis("off")
        ax.text(0.5, 0.5, f"{title}: no rows to show", ha="center", va="center", color=INK_SECONDARY)
        return _save(fig, path)
    length = max(len(row["images"]) for row in rows)
    fig, axes = plt.subplots(len(rows), length, figsize=(1.0 * length + 2.6, 1.08 * len(rows) + 0.6), squeeze=False)
    for r, row in enumerate(rows):
        for c in range(length):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(GRID)
            if c >= len(row["images"]):
                ax.axis("off")
                continue
            if row["images"][c] is not None:
                ax.imshow(row["images"][c], cmap="gray", vmin=pixel_range[0], vmax=pixel_range[1], interpolation="nearest")
            else:
                ax.text(0.5, 0.45, row["tokens"][c], ha="center", va="center", fontsize=20, color=INK, transform=ax.transAxes)
                ax.text(0.5, 0.06, "text", ha="center", va="bottom", fontsize=6, color=INK_MUTED, transform=ax.transAxes)
        caption = f"sum {row['target']}"
        if row.get("prediction") is not None:
            correct = row["prediction"] == row["target"]
            caption += f"\npred {row['prediction']} {'✓' if correct else '✗'}"
        if row.get("caption"):
            caption += f"\n{row['caption']}"
        color = INK if row.get("prediction") is None else (GOOD if row["prediction"] == row["target"] else CRITICAL)
        axes[r, -1].text(1.12, 0.5, caption, transform=axes[r, -1].transAxes, va="center", ha="left", fontsize=8.5, color=color)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.92, 0.97))
    return _save(fig, path)


def manifest_gallery_rows(manifest, partition_images: np.ndarray, row_indices, predictions=None, captions=None) -> list[dict]:
    """Gallery rows showing *raw* pixels for image slots and the digit string for text slots.

    Text strings come from the manifest digits through the tokenizer's number_to_string, the same
    conversion the dataset uses.
    """
    from tokenizer import number_to_string

    rows = []
    for k, index in enumerate(row_indices):
        images, tokens = [], []
        for digit, is_image, source in zip(manifest.digits[index], manifest.is_image[index], manifest.source_index[index]):
            images.append(partition_images[source] if is_image else None)
            tokens.append(None if is_image else number_to_string(int(digit)))
        rows.append(
            {
                "images": images,
                "tokens": tokens,
                "target": int(manifest.targets[index]),
                "prediction": None if predictions is None else int(predictions[k]),
                "caption": None if captions is None else captions[k],
            }
        )
    return rows


def plot_augmentation_preview(columns: dict[str, list[list[np.ndarray]]], path: Path, title: str) -> Path:
    """Rows = digits 0-9. Column groups = {mode name: per-digit list of uint8 images}."""
    groups = list(columns.items())
    total_columns = sum(len(images[0]) for _, images in groups)
    fig, axes = plt.subplots(10, total_columns, figsize=(0.62 * total_columns + 0.6, 7.0), squeeze=False)
    start = 0
    for group_name, per_digit in groups:
        width = len(per_digit[0])
        for digit in range(10):
            for k in range(width):
                ax = axes[digit, start + k]
                ax.imshow(per_digit[digit][k], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color(GRID)
                if k == 0 and start == 0:
                    ax.set_ylabel(str(digit), rotation=0, labelpad=10, color=INK_SECONDARY, fontsize=9, va="center")
        axes[0, start].set_title(group_name, loc="left", fontsize=9, color=INK_SECONDARY)
        start += width
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, path)


# ----------------------------------------------------------------------------- distributions


def plot_distributions(
    targets_by_split: dict[str, np.ndarray],
    image_counts_by_split: dict[str, np.ndarray],
    exact_sum_pmf: np.ndarray,
    image_count_pmf: np.ndarray,
    path: Path,
) -> Path:
    """Left: target-sum frequency per split vs the exact distribution of a sum of uniform digits.
    Right: image-operand count per sequence vs Binomial(L, p)."""
    fig, (sum_ax, count_ax) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    classes = np.arange(len(exact_sum_pmf))
    for split, targets in targets_by_split.items():
        frequency = np.bincount(targets, minlength=len(classes)) / len(targets)
        sum_ax.plot(classes, frequency, color=SERIES[split], linewidth=1.6, label=f"{split} (n={len(targets):,})")
    sum_ax.plot(classes, exact_sum_pmf, color=INK, linewidth=1.2, linestyle="--", label="exact pmf")
    sum_ax.set_xlabel("Target sum")
    sum_ax.set_ylabel("Fraction of sequences")
    sum_ax.legend(loc="upper right")
    _style(sum_ax)

    counts = np.arange(len(image_count_pmf))
    splits = list(image_counts_by_split)
    bar_width = 0.8 / len(splits)
    for k, split in enumerate(splits):
        observed = np.bincount(image_counts_by_split[split], minlength=len(counts)) / len(image_counts_by_split[split])
        offsets = counts - 0.4 + bar_width * (k + 0.5)
        count_ax.bar(offsets, observed, width=bar_width * 0.88, color=SERIES[split], label=split)
    count_ax.plot(counts, image_count_pmf, color=INK, marker="o", markersize=3, linewidth=1.0, linestyle="--", label="Binomial pmf")
    count_ax.set_xticks(counts)
    count_ax.set_xlabel("Image operands per sequence")
    count_ax.set_ylabel("Fraction of sequences")
    count_ax.legend(loc="upper right")
    _style(count_ax)
    fig.tight_layout()
    return _save(fig, path)


def plot_breakdown(by_image_count: list[dict], by_sum_range: list[dict], path: Path, title: str) -> Path:
    """Accuracy by number of image operands and by target-sum range, annotated with n."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for ax, rows, label in ((axes[0], by_image_count, "Image operands"), (axes[1], by_sum_range, "Target-sum range")):
        names = [str(row["group"]) for row in rows]
        values = [row["accuracy"] if row["n"] else 0.0 for row in rows]
        positions = np.arange(len(rows))
        ax.bar(positions, values, width=0.82, color=SERIES["test"] if "test" in title.lower() else SERIES["val"])
        for x, row in zip(positions, rows):
            text = f"{row['accuracy']:.2f}\nn={row['n']}" if row["n"] else "n=0"
            ax.text(x, (row["accuracy"] if row["n"] else 0) + 0.02, text, ha="center", va="bottom", fontsize=6.5, color=INK_SECONDARY)
        ax.set_xticks(positions, names, rotation=0 if len(rows) <= 11 else 45)
        ax.set_ylim(0, 1.18)
        ax.set_xlabel(label)
        ax.set_ylabel("Exact-sum accuracy")
        _style(ax)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, path)
