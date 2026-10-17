"""Explicit length-generalization evaluation by free generation. Run test only when you mean it.

    python evaluate_bonus.py --checkpoint runs/<run>/checkpoints/best.pt --split val
    python evaluate_bonus.py --checkpoint runs/<run>/checkpoints/best.pt --split test

Every answer is generated from BOS using the model's own previous predictions (greedy, cap
bonus.max_new_tokens). Correct answer prefixes are never provided. Teacher-forced loss is
reported separately and is not an accuracy.

Writes <run>/bonus_eval_<split>_<checkpoint>/:
    metrics.json            everything below, with sample counts
    predictions.csv         one row per naturally sampled sequence
    accuracy_by_length.png  natural / unseen-input / ORACLE / all-image / recognition reference
    invalid_by_length.png   invalid-output and missing-EOS rates
    repeated_operands.png   [d]*k accuracy by k, text and image
    examples.png            seeded random natural rows (successes and failures alike)
    failures.png            seeded random natural errors only

Report sections (separate populations, never mixed):
    by_length               natural sequences, every length 1..max_eval_length
    aggregate               trained lengths vs extrapolation lengths
    sum_groups              <=90 / 91-99 / >=100, per length and pooled over extrapolation lengths
    by_image_fraction       0 / (0,1/4] / ... / 1 of the operands are images
    diagnostics             ORACLE text substitution, all-image sets, text-only / mixed rows,
                            permutation consistency, repeated operands, repeated zeros,
                            appended zero, structural checks, recognition reference

Interpretation guide:
  - Longer inputs, larger sums, and unseen three-digit answers are different difficulties. A
    length-11 row mostly has a two-digit sum the model has written many times; a length-20 row
    with sum >= 100 needs a third digit it has never produced. Training sums never exceed 90.
  - Sum groups condition on the total, which also changes the operands: sums >= 100 at a given
    length consist of larger digits than typical rows. They separate the difficulties only
    partly.
  - The model *accepts* any length (padding masks; positions computed on demand). That is
    implementation support. Whether it *generalizes* is what the numbers below measure.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

import visualization as viz
from bonus_checks import structural_report
from bonus_data import constructed_manifest, eval_lengths, manifest_name, train_lengths
from bonus_engine import (
    by_length,
    evaluation_dataset,
    evaluation_loader,
    load_bonus_data,
    load_bonus_model,
    score,
    summarize,
    teacher_forced_loss_of,
)
from config import config_to_dict
from dataset import SequenceManifest
from engine import resolve_device
from experiment_logging import flatten_metrics, log_evaluation
from tokenizer import number_to_string

SUM_GROUPS = (("<=90", 0, 90), ("91-99", 91, 99), (">=100", 100, 10**9))
IMAGE_FRACTION_BINS = (("0", 0.0, 0.0), ("(0,0.25]", 1e-9, 0.25), ("(0.25,0.5]", 0.25 + 1e-9, 0.5), ("(0.5,0.75]", 0.5 + 1e-9, 0.75), ("(0.75,1)", 0.75 + 1e-9, 1 - 1e-9), ("1", 1.0, 1.0))
CNN_RUNS = Path("CNN-encoder-test/runs")


def run_scored(model, cfg, data, manifests: list[SequenceManifest], device) -> tuple[dict, object]:
    dataset = evaluation_dataset(cfg, data, manifests)
    return score(model, evaluation_loader(cfg, dataset, device), device, cfg), dataset


def append_zero(manifest: SequenceManifest, as_image: bool, data, seed: int) -> SequenceManifest:
    """Same rows with one real zero operand appended (text "0", or an image of a 0). Sum unchanged."""
    rows = len(manifest.digits)
    zero_digits = np.concatenate([manifest.digits, np.zeros((rows, 1), np.uint8)], axis=1)
    zero_is_image = np.concatenate([manifest.is_image, np.full((rows, 1), as_image)], axis=1)
    source = np.concatenate([manifest.source_index, np.full((rows, 1), -1, np.int32)], axis=1)
    if as_image:
        from dataset import SPLIT_TO_PARTITION

        labels = data.arrays.labels[SPLIT_TO_PARTITION[manifest.pool_split]]
        pool = data.pools.ids[manifest.pool_split]
        zeros = pool[labels[pool] == 0]
        source[:, -1] = zeros[np.random.default_rng(seed).integers(0, len(zeros), rows)]
    return SequenceManifest(f"{manifest.name}+zero_{'image' if as_image else 'text'}", manifest.pool_split, zero_digits, zero_is_image, source, manifest.targets.copy(), {"derived_from": manifest.name})


def train_text_inputs(data, cfg) -> set:
    """All-text training inputs as (length, sorted digits): reorderings count as seen (conservative)."""
    seen = set()
    for L in train_lengths(cfg):
        m = data.manifests[manifest_name("train", L)]
        for digits, images in zip(m.digits.tolist(), m.is_image.tolist()):
            if not any(images):
                seen.add((L, tuple(sorted(digits))))
    return seen


def length_table(result: dict, natural_manifests: list, dataset, seen: set) -> list[dict]:
    """Per length: overall summary, unseen-inputs-only summary, and sum groups."""
    unseen = np.ones(len(result["target"]), dtype=bool)
    for i, global_row in enumerate(result["index"]):
        which, local = dataset.locate(int(global_row))
        m = natural_manifests[which]
        if not m.is_image[local].any() and (m.sequence_length, tuple(sorted(m.digits[local].tolist()))) in seen:
            unseen[i] = False
    table = []
    for row in by_length(result):
        L = row["group"]
        at_length = result["length"] == L
        row["unseen_inputs"] = summarize(result, at_length & unseen)
        row["sum_groups"] = [{"group": name, **summarize(result, at_length & (result["target"] >= lo) & (result["target"] <= hi))} for name, lo, hi in SUM_GROUPS]
        table.append(row)
    return table, unseen


def image_fraction_table(result: dict, mask: np.ndarray) -> list[dict]:
    fraction = result["image_count"] / result["length"]
    return [{"group": name, **summarize(result, mask & (fraction >= lo) & (fraction <= hi))} for name, lo, hi in IMAGE_FRACTION_BINS]


def per_image_accuracy_reference(split: str, cfg) -> float | None:
    """Mean digit accuracy of CNN-encoder-test runs with this encoder configuration (same split), if any."""
    values = []
    for metrics_path in CNN_RUNS.glob("*/metrics.json"):
        metrics = json.loads(metrics_path.read_text())
        encoder = metrics["settings"].get("encoder") or {
            "activation": metrics["settings"].get("activation") or "gelu",
            "pooling": metrics["settings"]["model"]["cnn_pooling"],
            "batch_norm": bool(metrics["settings"].get("batch_norm")),
        }
        if (encoder["activation"], encoder["pooling"], encoder["batch_norm"]) == (cfg.model.cnn_activation, cfg.model.cnn_pooling, cfg.model.cnn_batch_norm):
            values.append(metrics[split]["accuracy"])
    return statistics.mean(values) if values else None


def gallery_rows(result: dict, positions: np.ndarray, manifests: list, dataset, data) -> list[dict]:
    rows = []
    for i in positions:
        which, local = dataset.locate(int(result["index"][i]))
        manifest = manifests[which]
        row = viz.manifest_gallery_rows(manifest, data.arrays.images[manifest.partition], [local])[0]
        text = result["text"][i] if result["status"][i] == "valid" else f"<{result['status'][i]}>"
        row.update(prediction_text=f"'{text}'", correct=bool(result["correct"][i]), caption=f"L={int(result['length'][i])}")
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=["disabled", "offline", "online"])
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    model, cfg, checkpoint = load_bonus_model(checkpoint_path, device)
    model.eval()
    data = load_bonus_data(cfg)
    current, stored = data.data_ids(), checkpoint["data_ids"]
    if current["source_pool_id"] != stored["source_pool_id"] or any(current["manifest_ids"].get(k) != v for k, v in stored["manifest_ids"].items()):
        raise ValueError("Bonus pools/manifests on disk differ from those this checkpoint was trained with")
    split, lengths, max_train = args.split, eval_lengths(cfg), cfg.bonus.max_train_length
    out = Path(args.out) if args.out else checkpoint_path.resolve().parent.parent / f"bonus_eval_{split}_{checkpoint_path.stem}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{checkpoint['architecture']} | step {checkpoint['train_state']['global_step']} | selection: {checkpoint['selection_rule']}")
    print(f"Evaluating split={split}, lengths 1-{lengths[-1]} (trained 1-{max_train}), free generation, cap {cfg.bonus.max_new_tokens} tokens, {device}")

    # Naturally sampled sequences.
    natural_manifests = [data.manifests[manifest_name(split, L)] for L in lengths]
    natural, natural_set = run_scored(model, cfg, data, natural_manifests, device)
    table, unseen = length_table(natural, natural_manifests, natural_set, train_text_inputs(data, cfg))
    trained, extrapolation = natural["length"] <= max_train, natural["length"] > max_train
    aggregate = {
        f"trained_lengths_1_{max_train}": {**summarize(natural, trained), "tf_loss": teacher_forced_loss_of(natural, trained)},
        f"extrapolation_lengths_{max_train + 1}_{lengths[-1]}": {**summarize(natural, extrapolation), "tf_loss": teacher_forced_loss_of(natural, extrapolation)},
        "extrapolation_unseen_inputs": summarize(natural, extrapolation & unseen),
    }
    extrapolation_sum_groups = [{"group": name, **summarize(natural, extrapolation & (natural["target"] >= lo) & (natural["target"] <= hi))} for name, lo, hi in SUM_GROUPS]
    image_fraction = {"trained_lengths": image_fraction_table(natural, trained), "extrapolation_lengths": image_fraction_table(natural, extrapolation)}

    # Diagnostics.
    oracle, _ = run_scored(model, cfg, data, [m.with_all_text() for m in natural_manifests], device)
    all_image, _ = run_scored(model, cfg, data, [data.manifests[manifest_name(split, L, "all_image")] for L in lengths], device)
    permuted, _ = run_scored(model, cfg, data, [m.permuted(cfg.eval.permutation_seed + L) for L, m in zip(lengths, natural_manifests)], device)
    same_answer = (permuted["text"] == natural["text"]) & (permuted["status"] == natural["status"])
    zero_bases = natural_manifests[:-1]  # appending makes length L+1, so stop one short of the eval range
    zero_results = {}
    for as_image in (False, True):
        result, _ = run_scored(model, cfg, data, [append_zero(m, as_image, data, cfg.eval.stress_seed + i) for i, m in enumerate(zero_bases)], device)
        base_mask = natural["length"] < lengths[-1]
        agreement = (result["text"] == natural["text"][base_mask]) & (result["status"] == natural["status"][base_mask])
        zero_results["image" if as_image else "text"] = {
            "by_original_length": [
                {"group": L, "n": int((natural["length"][base_mask] == L).sum()),
                 "answer_unchanged_rate": float(agreement[natural["length"][base_mask] == L].mean()),
                 "accuracy_after_append": float(result["correct"][natural["length"][base_mask] == L].mean())}
                for L in lengths[:-1]
            ]
        }
    repeated = {}
    for as_image in (False, True):
        manifests = [constructed_manifest(f"repeat_k{k:02d}", split, [[d] * k for d in range(10)], [[as_image] * k for _ in range(10)], data.arrays, data.pools, cfg.eval.stress_seed + k) for k in lengths]
        result, _ = run_scored(model, cfg, data, manifests, device)
        digits = np.tile(np.arange(10), len(lengths))
        repeated["image" if as_image else "text"] = {
            "rows": [{"digit": int(d), "k": int(L), "target": int(t), "generated": text if status == "valid" else f"<{status}>", "correct": bool(c)}
                     for d, L, t, text, status, c in zip(digits, result["length"], result["target"], result["text"], result["status"], result["correct"])],
            "accuracy_by_k_digits_1_9": [{"group": int(L), **summarize(result, (result["length"] == L) & (digits > 0))} for L in lengths],
            "repeated_zeros_by_k": [{"group": int(L), **summarize(result, (result["length"] == L) & (digits == 0))} for L in lengths],
        }
    structure = structural_report(model, device, max_length=lengths[-1])
    p_image = per_image_accuracy_reference(split, cfg)
    q = cfg.data.image_probability
    reference = None if p_image is None else {
        "per_image_accuracy": p_image,
        "note": "Analytic reference (1 - q + q*p)^L: recognition-limited accuracy IF image errors were independent and arithmetic perfect. "
        "p = mean digit accuracy of CNN-encoder-test runs with this encoder configuration (same split). Not a model result.",
        "by_length": [{"group": L, "value": (1 - q + q * p_image) ** L} for L in lengths],
    }
    diagnostics = {
        "oracle_text_substitution": {"warning": "ORACLE: images replaced by their true digit strings via the tokenizer (uses labels; every row becomes all-text, unlike training).", "by_length": by_length(oracle)},
        "all_image_sets": {"by_length": by_length(all_image)},
        "natural_text_only_rows": {"by_length": by_length(natural, natural["image_count"] == 0)},
        "natural_mixed_rows": {"by_length": by_length(natural, (natural["image_count"] > 0) & (natural["image_count"] < natural["length"]))},
        "permutation_consistency": {
            "note": "Operands shuffled per row (seeded); the sum is unchanged. Agreement = identical generated output. Positional models are not invariant by construction.",
            "by_length": [{"group": L, "n": int((natural["length"] == L).sum()), "agreement": float(same_answer[natural["length"] == L].mean()),
                           "accuracy_shuffled": float(permuted["correct"][permuted["length"] == L].mean())} for L in lengths],
        },
        "append_zero": zero_results,
        "repeated_operands": repeated,
        "structural": structure,
        "recognition_reference": reference,
    }

    metrics = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "architecture": checkpoint["architecture"],
        "positions": checkpoint["positions"],
        "optimizer_step": checkpoint["train_state"]["global_step"],
        "selection_rule": checkpoint["selection_rule"],
        "generation": {"decoding": "greedy", "max_new_tokens": cfg.bonus.max_new_tokens, "special_token_suppression": False},
        "by_length": table,
        "aggregate": aggregate,
        "extrapolation_sum_groups": extrapolation_sum_groups,
        "by_image_fraction": image_fraction,
        "diagnostics": diagnostics,
        "notes": [
            "Implementation support for any length (padding masks, positions on demand) is not learned generalization; see by_length.",
            "Training sums are <= 90; answers >= 100 need a third digit the decoder never produced in training.",
            f"Checkpoint selected on validation lengths 1-{max_train} only; lengths {max_train + 1}-{lengths[-1]} are extrapolation.",
        ],
        "data_ids": current,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    with open(out / "predictions.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["length", "operands", "num_images", "target", "generated", "status", "correct", "unseen_input"])
        for i, global_row in enumerate(natural["index"]):
            which, local = natural_set.locate(int(global_row))
            m = natural_manifests[which]
            operands = " ".join(f"[{int(d)}]" if img else number_to_string(int(d)) for d, img in zip(m.digits[local], m.is_image[local]))
            writer.writerow([int(natural["length"][i]), operands, int(natural["image_count"][i]), int(natural["target"][i]), natural["text"][i], natural["status"][i], int(natural["correct"][i]), int(unseen[i])])

    counts = {row["group"]: row["n"] for row in table}
    series = {
        "natural (free generation)": [(r["group"], r["accuracy"]) for r in table],
        "unseen inputs only": [(r["group"], r["unseen_inputs"]["accuracy"]) for r in table],
        "ORACLE text substitution": [(r["group"], r["accuracy"]) for r in by_length(oracle)],
        "all-image set": [(r["group"], r["accuracy"]) for r in by_length(all_image)],
    }
    if reference:
        series["recognition reference (analytic)"] = [(r["group"], r["value"]) for r in reference["by_length"]]
    figures = {
        "accuracy_by_length": viz.plot_by_length(series, out / "accuracy_by_length.png", f"{split}: exact-answer accuracy by length ({checkpoint['architecture']})", "Exact full-answer accuracy", max_train, counts),
        "invalid_by_length": viz.plot_by_length(
            {"invalid output": [(r["group"], r["invalid_rate"]) for r in table], "missing EOS": [(r["group"], r["missing_eos_rate"]) for r in table]},
            out / "invalid_by_length.png", f"{split}: invalid-output rates by length", "Fraction of rows", max_train, counts,
        ),
        "repeated_operands": viz.plot_by_length(
            {f"[d]*k as {kind} (d=1..9)": [(r["group"], r["accuracy"]) for r in repeated[kind]["accuracy_by_k_digits_1_9"]] for kind in ("text", "image")}
            | {"[0]*k as text -> '0'": [(r["group"], r["accuracy"]) for r in repeated["text"]["repeated_zeros_by_k"]]},
            out / "repeated_operands.png", f"{split}: repeated operands [d, d, ..., d] (k copies)", "Exact-answer accuracy", max_train,
        ),
    }
    rng = np.random.default_rng(cfg.eval.gallery_seed)
    picks = np.concatenate([rng.choice(np.flatnonzero(natural["length"] == L), size=min(2, int((natural["length"] == L).sum())), replace=False) for L in (1, 5, max_train, max_train + 1, (max_train + lengths[-1]) // 2, lengths[-1]) if (natural["length"] == L).any()])
    figures["examples"] = viz.plot_sequence_gallery(gallery_rows(natural, picks, natural_manifests, natural_set, data), out / "examples.png", f"{split}: seeded random rows at selected lengths (successes and failures)")
    errors = np.flatnonzero(~natural["correct"])
    fail_picks = np.sort(np.random.default_rng(cfg.eval.gallery_seed + 1).choice(errors, size=min(16, len(errors)), replace=False)) if len(errors) else np.array([], dtype=int)
    figures["failures"] = viz.plot_sequence_gallery(gallery_rows(natural, fail_picks, natural_manifests, natural_set, data), out / "failures.png", f"{split}: failure gallery, {len(fail_picks)} random of {len(errors)} errors")

    run_name = checkpoint_path.resolve().parent.parent.name
    logging_cfg = cfg.logging if args.wandb_mode is None else dataclasses.replace(cfg.logging, wandb_mode=args.wandb_mode)
    scalars = {key: metrics[key] for key in ("by_length", "aggregate", "extrapolation_sum_groups", "by_image_fraction")}
    log_evaluation(
        logging_cfg,
        run_config={**config_to_dict(cfg), "evaluation": {"checkpoint": str(checkpoint_path), "split": split, "task": "bonus_decoder"}},
        summary={**flatten_metrics(scalars, prefix=f"bonus_{split}"), "optimizer_step": metrics["optimizer_step"]},
        images={f"bonus_{split}/{name}": path for name, path in figures.items()},
        name=f"{run_name}-{out.name}",
        group=run_name,
        directory=out,
    )

    print(f"\n{'L':>3} {'n':>6} {'exact acc':>9} {'unseen acc (n)':>15} {'invalid':>8} {'no EOS':>7} {'MAE valid (n)':>15} {'<=90':>11} {'91-99':>11} {'>=100':>11}")
    for row in table:
        groups = {g["group"]: g for g in row["sum_groups"]}
        fmt = lambda g: f"{g['accuracy']:.2f}({g['n']})" if g["n"] else "-"  # noqa: E731
        unseen_text = f"{row['unseen_inputs']['accuracy']:.3f}({row['unseen_inputs']['n']})" if row["unseen_inputs"]["n"] else "-"
        mae = f"{row['mae_valid']:.2f}({row['n_valid']})" if row["mae_valid"] is not None else f"-({row['n_valid']})"
        marker = "" if row["group"] <= max_train else " *"
        print(f"{row['group']:>3} {row['n']:>6} {row['accuracy']:>9.3f} {unseen_text:>15} {row['invalid_rate']:>8.3f} {row['missing_eos_rate']:>7.3f} {mae:>15} "
              f"{fmt(groups['<=90']):>11} {fmt(groups['91-99']):>11} {fmt(groups['>=100']):>11}{marker}")
    print("(* = extrapolation length)")
    for name, value in aggregate.items():
        print(f"{name:<34} acc {value['accuracy']:.4f} (n={value['n']})" + (f"  tf-loss {value['tf_loss']:.4f}" if value.get("tf_loss") is not None else ""))
    print("extrapolation by sum group: " + ", ".join(f"{g['group']}: {g['accuracy']:.3f} (n={g['n']})" if g["n"] else f"{g['group']}: - (n=0)" for g in extrapolation_sum_groups))
    print(f"structural: padding max |dlogit| {structure['padding_max_abs_logit_diff']:.1e}; [3]*k dist diff vs k=1 {structure['repeated_3_max_dist_diff_vs_k1[BOS]']:.1e}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
