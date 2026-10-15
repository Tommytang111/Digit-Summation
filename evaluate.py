"""Explicit checkpoint evaluation with diagnostics. Test is evaluated only when you ask for it.

    python evaluate.py --checkpoint runs/<run>/checkpoints/best.pt --split val
    python evaluate.py --checkpoint runs/<run>/checkpoints/best.pt --split test

Writes to <run>/eval_<split>_<checkpoint>/:
    metrics.json      main metrics, breakdowns (with n), baseline, diagnostics, stress cases
    predictions.csv   one row per sequence; answers as text
    examples.png      seeded random rows (successes and failures alike)
    failures.png      seeded random rows among the errors only
    breakdown.png     accuracy by image count and by sum range
    stress.png        all-zero / all-nine stress inputs (kept apart from natural results)

Report sections, kept separate on purpose:
    main         naturally sampled sequences of the split (the headline number)
    baseline     always answer the most frequent training sum
    diagnostics  text-only set, all-image set, permutation test, ORACLE substitution
    stress       hand-constructed extremes (sums 0 and 90 essentially never occur in training)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import visualization as viz
from augmentations import build_transform
from dataset import SequenceManifest, compute_targets, load_prepared_data
from engine import EvalOutput, evaluate, load_model_from_checkpoint, make_loader, resolve_device
from tokenizer import class_id_to_text, number_to_string


def run_eval(model, manifest: SequenceManifest, data, cfg, device, keep_logits=False) -> EvalOutput:
    dataset = data.dataset(manifest, build_transform(cfg.augment, train=False))  # deterministic preprocessing
    return evaluate(model, make_loader(dataset, cfg.eval.batch_size, device, cfg.train), device, cfg.train.amp_dtype, keep_logits)


def grouped(result: EvalOutput, groups: np.ndarray, labels: list) -> list[dict]:
    rows = []
    for label in labels:
        mask = groups == label if not isinstance(label, tuple) else (groups >= label[0]) & (groups <= label[1])
        n = int(mask.sum())
        correct = result.predictions[mask] == result.targets[mask]
        rows.append(
            {
                "group": label if not isinstance(label, tuple) else f"{label[0]}-{label[1]}",
                "n": n,
                "accuracy": float(correct.mean()) if n else None,
                "mae": float(np.abs(result.predictions[mask] - result.targets[mask]).mean()) if n else None,
            }
        )
    return rows


def stress_manifests(data, split: str, cfg) -> dict[str, SequenceManifest]:
    """All-zero / all-nine sequences as text, as images, and mixed. Not naturally sampled."""
    length = cfg.data.sequence_length
    count = cfg.eval.num_stress_image_sequences
    rng = np.random.default_rng(cfg.eval.stress_seed)
    partition = {"val": "train", "test": "test"}[split]
    labels = data.arrays.labels[partition]
    pool = data.pools.ids[split]
    manifests = {}
    for digit in (0, 9):
        candidates = pool[labels[pool] == digit]
        for kind, rows in (("text", 1), ("image", count), ("mixed", count)):
            digits = np.full((rows, length), digit, dtype=np.uint8)
            if kind == "text":
                is_image = np.zeros((rows, length), bool)
            elif kind == "image":
                is_image = np.ones((rows, length), bool)
            else:
                is_image = rng.random((rows, length)) < 0.5
            source = np.full((rows, length), -1, dtype=np.int32)
            source[is_image] = candidates[rng.integers(0, len(candidates), int(is_image.sum()))]
            targets = compute_targets(digits, is_image, source, labels).astype(np.int16)
            name = f"all_{digit}_{kind}"
            manifests[name] = SequenceManifest(name, split, digits, is_image, source, targets, {"stress": True})
    return manifests


def operand_string(manifest: SequenceManifest, row: int) -> str:
    """'5 [7] 3 ...': plain = text operand, [d] = image whose MNIST label is d."""
    parts = []
    for digit, is_image in zip(manifest.digits[row], manifest.is_image[row]):
        parts.append(f"[{int(digit)}]" if is_image else number_to_string(int(digit)))
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    model, cfg, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    data = load_prepared_data(cfg.data)
    stored_ids = checkpoint["data_ids"]
    current_ids = data.data_ids()
    for name in ("train", args.split, f"{args.split}_text_only", f"{args.split}_all_image"):
        if stored_ids["manifest_ids"].get(name) != current_ids["manifest_ids"].get(name):
            raise ValueError(f"Manifest {name!r} on disk differs from the one this checkpoint was trained with")
    if stored_ids["source_pool_id"] != current_ids["source_pool_id"]:
        raise ValueError("Source pools on disk differ from the checkpoint's")

    out = Path(args.out) if args.out else checkpoint_path.resolve().parent.parent / f"eval_{args.split}_{checkpoint_path.stem}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = data.manifests[args.split]
    partition_images = data.arrays.images[manifest.partition]
    train_state = checkpoint["train_state"]
    print(f"Checkpoint {checkpoint_path} (optimizer step {train_state['global_step']}, best val acc {train_state['best_val_accuracy']:.4f})")
    print(f"Evaluating split={args.split} ({manifest.num_sequences} sequences) on {device}")

    # Main, naturally sampled results.
    main_result = run_eval(model, manifest, data, cfg, device, keep_logits=True)
    image_counts = manifest.is_image.sum(axis=1)[main_result.indices]
    by_images = grouped(main_result, image_counts, list(range(cfg.data.sequence_length + 1)))
    max_sum = 9 * cfg.data.sequence_length
    ranges = [(low, min(low + 9, max_sum)) for low in range(0, max_sum + 1, 10)]
    if ranges[-1][0] == ranges[-1][1] and len(ranges) > 1:  # fold a lone top value (e.g. 90) into the previous bin
        ranges[-2] = (ranges[-2][0], ranges[-1][1])
        ranges.pop()
    by_range = grouped(main_result, main_result.targets, ranges)

    train_targets = data.manifests["train"].targets.astype(np.int64)
    train_counts = np.bincount(train_targets, minlength=cfg.num_classes)
    majority = int(train_counts.argmax())
    unseen = train_counts[main_result.targets] == 0
    baseline = {
        "rule": f"always answer the most frequent training sum ({majority})",
        "answer": class_id_to_text(majority),
        "accuracy": float((main_result.targets == majority).mean()),
        "mae": float(np.abs(main_result.targets - majority).mean()),
    }

    # Diagnostics (each its own population; not mixed into the headline).
    text_only = run_eval(model, data.manifests[f"{args.split}_text_only"], data, cfg, device)
    all_image = run_eval(model, data.manifests[f"{args.split}_all_image"], data, cfg, device)
    permuted = run_eval(model, manifest.permuted(cfg.eval.permutation_seed), data, cfg, device, keep_logits=True)
    oracle = run_eval(model, manifest.with_all_text(), data, cfg, device)
    diagnostics = {
        "text_only_set": text_only.metrics(),
        "all_image_set": all_image.metrics(),
        "permutation_test": {
            **permuted.metrics(),
            "prediction_agreement_with_original_order": float((permuted.predictions == main_result.predictions).mean()),
            "max_abs_logit_difference": float(np.abs(permuted.logits - main_result.logits).max()),
        },
        "oracle_substitution": {
            **oracle.metrics(),
            "warning": "ORACLE: images replaced by their true digit tokens (uses ground-truth labels). "
            "Every row becomes all-text, a modality mix that is rare in training.",
        },
        "rows_with_sum_unseen_in_training": {
            "n": int(unseen.sum()),
            "accuracy": float((main_result.predictions[unseen] == main_result.targets[unseen]).mean()) if unseen.any() else None,
        },
    }

    stress = {}
    stress_rows = []
    for name, stress_manifest in stress_manifests(data, args.split, cfg).items():
        result = run_eval(model, stress_manifest, data, cfg, device)
        stress[name] = {**result.metrics(), "target": int(stress_manifest.targets[0]), "predictions_seen": sorted(set(result.predictions.tolist()))[:10]}
        stress_rows += viz.manifest_gallery_rows(stress_manifest, partition_images, [0], result.predictions[:1], captions=[name])

    metrics = {
        "checkpoint": str(checkpoint_path),
        "optimizer_step": train_state["global_step"],
        "split": args.split,
        "main": {**main_result.metrics(), "by_image_count": by_images, "by_sum_range": by_range},
        "majority_baseline": baseline,
        "diagnostics": diagnostics,
        "stress_cases": stress,
        "data_ids": current_ids,
    }
    with open(out / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)

    with open(out / "predictions.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "operands", "num_images", "target", "prediction_text", "correct", "abs_error"])
        for row, target, prediction in zip(main_result.indices, main_result.targets, main_result.predictions):
            writer.writerow(
                [int(row), operand_string(manifest, int(row)), int(manifest.is_image[row].sum()), int(target),
                 class_id_to_text(int(prediction)), int(prediction == target), int(abs(int(prediction) - int(target)))]
            )

    k = cfg.eval.num_gallery_examples
    rng = np.random.default_rng(cfg.eval.gallery_seed)
    shown = np.sort(rng.choice(len(main_result.targets), size=min(k, len(main_result.targets)), replace=False))
    viz.plot_sequence_gallery(
        viz.manifest_gallery_rows(manifest, partition_images, main_result.indices[shown], main_result.predictions[shown]),
        out / "examples.png",
        f"{args.split}: seeded random rows (seed {cfg.eval.gallery_seed}), successes and failures alike",
    )
    failures = np.flatnonzero(main_result.predictions != main_result.targets)
    picked = np.sort(np.random.default_rng(cfg.eval.gallery_seed + 1).choice(failures, size=min(k, len(failures)), replace=False)) if len(failures) else []
    viz.plot_sequence_gallery(
        viz.manifest_gallery_rows(manifest, partition_images, main_result.indices[picked], main_result.predictions[picked]) if len(picked) else [],
        out / "failures.png",
        f"{args.split}: failure gallery, {len(picked)} random of {len(failures)} errors",
    )
    viz.plot_breakdown(by_images, by_range, out / "breakdown.png", f"{args.split}: accuracy breakdown (n per bar)")
    viz.plot_sequence_gallery(stress_rows, out / "stress.png", "Stress cases (constructed; not part of the natural results)")

    print(f"\n== {args.split} (naturally sampled) ==")
    print(f"exact-sum accuracy {main_result.accuracy:.4f}  MAE {main_result.mae:.3f}  loss {main_result.loss:.4f}  n={main_result.num_examples}")
    print(f"majority baseline ({baseline['answer']}): accuracy {baseline['accuracy']:.4f}  MAE {baseline['mae']:.3f}")
    print("by image count: " + ", ".join(f"{r['group']}:{r['accuracy']:.2f}(n={r['n']})" for r in by_images if r["n"]))
    print("by sum range:   " + ", ".join(f"{r['group']}:{r['accuracy']:.2f}(n={r['n']})" for r in by_range if r["n"]))
    print("\n== diagnostics (separate populations) ==")
    print(f"text-only set  acc {text_only.accuracy:.4f} (n={text_only.num_examples})")
    print(f"all-image set  acc {all_image.accuracy:.4f} (n={all_image.num_examples})")
    p = diagnostics["permutation_test"]
    print(f"permutation    acc {p['accuracy']:.4f}; agreement {p['prediction_agreement_with_original_order']:.4f}; max |dlogit| {p['max_abs_logit_difference']:.2e}")
    print(f"ORACLE text substitution acc {oracle.accuracy:.4f} (uses true labels; changes modality mix)")
    print("\n== stress cases (constructed) ==")
    for name, s in stress.items():
        print(f"{name:>14}: target {s['target']:>2}  acc {s['accuracy']:.2f}  (n={s['n']}, predictions {s['predictions_seen']})")
    example = int(main_result.indices[0])
    print(f"\nexample answer: operands {operand_string(manifest, example)} -> \"{class_id_to_text(int(main_result.predictions[0]))}\" (target {int(main_result.targets[0])})")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
