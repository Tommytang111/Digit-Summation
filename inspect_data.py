"""Inspect prepared data before training: examples, augmentations, tokenization, targets, stats.

    python inspect_data.py --config configs/debug.yaml            # writes runs/inspect-<name>/

Outputs:
    raw_examples.png          training sequences with raw pixels + digit strings + sums
    model_view_examples.png   the same rows after the training transform (normalized pixels)
    augmentation_preview.png  every digit 0-9: raw | affine samples | affine_elastic samples
    distributions.png         sum and image-count distributions per split vs exact pmfs
    inspection_summary.json   pixel ranges, normalization check, elastic displacement, stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import albumentations as A
import numpy as np
import torch

import visualization as viz
from augmentations import (
    MNIST_MEAN,
    MNIST_STD,
    NORMALIZED_BLACK,
    NORMALIZED_WHITE,
    build_transform,
    geometric_transforms,
    measure_elastic_displacement,
)
from config import load_config
from dataset import collate_batch, load_prepared_data
from tokenizer import encode_operand, number_to_string


def exact_sum_pmf(length: int) -> np.ndarray:
    """Exact distribution of the sum of `length` independent Uniform{0..9} digits (convolution)."""
    pmf = np.array([1.0])
    for _ in range(length):
        pmf = np.convolve(pmf, np.full(10, 0.1))
    return pmf


def binomial_pmf(n: int, p: float) -> np.ndarray:
    from math import comb

    return np.array([comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)])


def trace_row(manifest, labels: np.ndarray, row: int) -> list[str]:
    """Human-readable trace: modality, value, string, token ID, and the target computation."""
    lines = [f"{'pos':>3} {'modality':>8} {'value':>5} {'string':>6} {'token_id':>8}  source"]
    terms = []
    for position, (digit, is_image, source) in enumerate(
        zip(manifest.digits[row], manifest.is_image[row], manifest.source_index[row])
    ):
        if is_image:
            label = int(labels[source])
            lines.append(f"{position:>3} {'image':>8} {label:>5} {'-':>6} {'PAD(0)':>8}  MNIST[{manifest.partition}][{source}] label={label}")
            terms.append(f"[img:{label}]")
        else:
            text = number_to_string(int(digit))
            lines.append(f"{position:>3} {'text':>8} {int(digit):>5} {text!r:>6} {encode_operand(int(digit)):>8}")
            terms.append(text)
    lines.append(f"target = {' + '.join(terms)} = {int(manifest.targets[row])}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", default=None)
    parser.add_argument("--rows", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.out or f"runs/inspect-{cfg.name}")
    out.mkdir(parents=True, exist_ok=True)
    data = load_prepared_data(cfg.data)
    train_manifest = data.manifests["train"]
    train_images = data.arrays.images["train"]
    train_labels = data.arrays.labels["train"]
    summary: dict = {"config": cfg.name, "augment_mode": cfg.augment.mode}

    # 1. Raw examples + tokenization/target trace.
    rows = np.sort(np.random.default_rng(0).choice(train_manifest.num_sequences, size=args.rows, replace=False))
    viz.plot_sequence_gallery(
        viz.manifest_gallery_rows(train_manifest, train_images, rows, captions=[f"row {r}" for r in rows]),
        out / "raw_examples.png",
        "Training sequences: raw MNIST pixels and digit-string operands",
    )
    print("== Tokenization and target trace (training row %d) ==" % rows[0])
    trace = trace_row(train_manifest, train_labels, int(rows[0]))
    print("\n".join(trace))
    summary["trace_example"] = trace

    # 2. Model view: transformed pixels and batch tensor contract.
    train_transform = build_transform(cfg.augment, train=True, seed=123)
    dataset = data.dataset("train", train_transform)
    items = [dataset[int(r)] for r in rows]
    viz.plot_sequence_gallery(
        [
            {
                "images": [it["images"][p, 0].numpy() if it["is_image"][p] else None for p in range(len(it["is_image"]))],
                "tokens": [None if it["is_image"][p] else number_to_string(int(it["digits"][p])) for p in range(len(it["is_image"]))],
                "target": it["target"],
                "caption": f"row {r}",
            }
            for it, r in zip(items, rows)
        ],
        out / "model_view_examples.png",
        f"Same rows after the training transform (augment={cfg.augment.mode}); pixels normalized",
        pixel_range=(NORMALIZED_BLACK, NORMALIZED_WHITE),
    )
    batch = collate_batch(items)
    contract = {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in batch.items()}
    print("\n== Batch tensor contract ==")
    for key, value in contract.items():
        print(f"{key:>13}: {value['dtype']:<14} {value['shape']}")
    image_pixels = batch["images"][batch["is_image"]]
    summary["batch_contract"] = contract
    summary["pixel_checks"] = {
        "raw_dtype": str(train_images.dtype),
        "raw_range": [int(train_images.min()), int(train_images.max())],
        "model_input_range": [float(image_pixels.min()), float(image_pixels.max())],
        "expected_normalized_range": [NORMALIZED_BLACK, NORMALIZED_WHITE],
        "placeholder_values_at_text_slots": sorted(set(batch["images"][~batch["is_image"]].flatten().tolist())),
        "token_ids_at_image_slots": sorted(set(batch["token_ids"][batch["is_image"]].tolist())),
        "padding_mask_any": bool(batch["padding_mask"].any()),
        "any_float64": any(v.dtype == torch.float64 for v in batch.values()),
    }

    # 3. Normalization applied exactly once: deterministic transform statistics on the train pool.
    eval_transform = build_transform(cfg.augment, train=False)
    pool = data.pools.ids["train"]
    normalized = np.stack([eval_transform(train_images[i]) for i in pool])
    summary["normalization"] = {
        "constants": {"mean": MNIST_MEAN, "std": MNIST_STD},
        "train_pool_raw_mean_std": [float(train_images[pool].mean() / 255), float(train_images[pool].std() / 255)],
        "train_pool_normalized_mean_std": [float(normalized.mean()), float(normalized.std())],
        "black_pixel_maps_to": float(eval_transform(np.zeros((28, 28), np.uint8))[0, 0]),
        "white_pixel_maps_to": float(eval_transform(np.full((28, 28), 255, np.uint8))[0, 0]),
    }
    print("\n== Normalization ==")
    print(json.dumps(summary["normalization"], indent=2))

    # 4. Augmentation preview over all ten digits.
    preview_rng = np.random.default_rng(1)
    examples_per_digit = [pool[train_labels[pool] == d][preview_rng.integers(0, (train_labels[pool] == d).sum())] for d in range(10)]
    columns = {"raw": [[train_images[i]] for i in examples_per_digit]}
    for mode in ("affine", "affine_elastic"):
        pipeline = A.Compose(geometric_transforms(cfg.augment, mode), seed=7)  # uint8 ops only, no Normalize
        columns[mode] = [[pipeline(image=train_images[i])["image"] for _ in range(5)] for i in examples_per_digit]
    viz.plot_augmentation_preview(
        columns,
        out / "augmentation_preview.png",
        f"Augmentation preview (uint8, before normalization): rotate ±{cfg.augment.rotate_degrees}°, "
        f"translate ±{cfg.augment.translate_px}px, elastic α={cfg.augment.elastic_alpha} σ={cfg.augment.elastic_sigma}",
    )

    # 5. Elastic displacement measured with the installed library (pixels).
    calibration = [measure_elastic_displacement(alpha, cfg.augment.elastic_sigma) for alpha in (4, 8, cfg.augment.elastic_alpha, 20, 34)]
    summary["elastic_displacement_px"] = calibration
    print("\n== Elastic displacement (albumentations 2.0.8, uniform noise; pixels, interior only) ==")
    for row in calibration:
        print(
            f"alpha {row['alpha']:>5.1f} sigma {row['sigma']:.1f}: RMS/axis {row['rms_per_axis_px']:.2f}  "
            f"p95 |d| {row['p95_magnitude_px']:.2f}  max |d| {row['max_magnitude_px']:.2f}  "
            f"sampled outside image {100 * row['fraction_sampled_outside']:.2f}%"
        )

    # 6. Distributions vs exact expectations (do not assume uniform target classes).
    length = cfg.data.sequence_length
    pmf = exact_sum_pmf(length)
    targets = {split: data.manifests[split].targets.astype(np.int64) for split in ("train", "val", "test")}
    image_counts = {split: data.manifests[split].is_image.sum(axis=1) for split in ("train", "val", "test")}
    viz.plot_distributions(targets, image_counts, pmf, binomial_pmf(length, cfg.data.image_probability), out / "distributions.png")
    train_counts = np.bincount(targets["train"], minlength=len(pmf))
    summary["distributions"] = {
        "exact_mode": int(pmf.argmax()),
        "exact_mode_probability": float(pmf.max()),
        "exact_probability_sum_0_or_90": float(pmf[0] + pmf[-1]),
        "train_majority_sum": int(train_counts.argmax()),
        "train_majority_fraction": float(train_counts.max() / train_counts.sum()),
        "train_sums_never_seen": [int(k) for k in np.flatnonzero(train_counts == 0)],
        "mean_image_operands": {split: float(c.mean()) for split, c in image_counts.items()},
    }
    print("\n== Distributions ==")
    d = summary["distributions"]
    print(f"exact sum pmf peaks at {d['exact_mode']} with p={d['exact_mode_probability']:.4f}; P(sum in {{0, 90}}) = {d['exact_probability_sum_0_or_90']:.1e}")
    print(f"train majority sum {d['train_majority_sum']} ({100 * d['train_majority_fraction']:.2f}% of train)")
    print(f"sums absent from train: {len(d['train_sums_never_seen'])} classes, e.g. {d['train_sums_never_seen'][:6]} ... {d['train_sums_never_seen'][-6:]}")
    print(f"mean image operands per sequence: {d['mean_image_operands']}")

    with open(out / "inspection_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nWrote figures and inspection_summary.json to {out}")


if __name__ == "__main__":
    main()
