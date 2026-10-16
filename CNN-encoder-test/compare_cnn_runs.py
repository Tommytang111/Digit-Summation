"""Compare digit-CNN runs, grouped by encoder variant, with mean ± sample std over seeds.

    ../.torch/bin/python CNN-encoder-test/compare_cnn_runs.py                    # every run in runs/
    ../.torch/bin/python CNN-encoder-test/compare_cnn_runs.py digit-cnn-s0 relu-max-bn-s0

A variant is (activation, pooling, batch_norm, augment mode, epochs, batch size). Runs that
differ only in train.seed are grouped. Make decisions on validation numbers; test is reported
for reference. For one run, the test-accuracy sampling noise alone is about
sqrt(p(1-p)/10000) ~ ±0.08 points at p = 0.993.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"


def describe(metrics: dict) -> tuple:
    s = metrics["settings"]
    encoder = s.get("encoder") or {  # runs before the encoder options lived in the config
        "activation": s.get("activation") or "gelu",
        "pooling": s["model"]["cnn_pooling"],
        "batch_norm": bool(s.get("batch_norm")),
    }
    return (
        encoder["activation"],
        encoder["pooling"],
        "BN" if encoder["batch_norm"] else "no BN",
        s["augment"]["mode"],
        f"{s['epochs']} ep",
        f"batch {s['batch_size']}",
    )


def seed_of(metrics: dict) -> str:
    """The train.seed override of the run, or the config's seed when none was given."""
    for override in metrics["settings"].get("overrides", []):
        if override.startswith("train.seed="):
            return override.split("=", 1)[1]
    return "config"


def mean_std(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


def main(names: list[str]) -> int:
    run_dirs = [RUNS / n for n in names] if names else sorted(p for p in RUNS.iterdir() if (p / "metrics.json").exists())
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for run_dir in run_dirs:
        metrics = json.loads((run_dir / "metrics.json").read_text())
        metrics["run"] = run_dir.name
        groups[describe(metrics)].append(metrics)
    for variant, runs in groups.items():
        seeds = ", ".join(seed_of(r) for r in runs)
        print(f"\n{' / '.join(variant)}   (n={len(runs)} runs: {', '.join(r['run'] for r in runs)}; seeds {seeds})")
        rows = {
            "val digit accuracy": [r["val"]["accuracy"] for r in runs],
            "val digit macro F1": [r["val"]["macro_f1"] for r in runs],
            "test digit accuracy": [r["test"]["accuracy"] for r in runs],
            "test digit macro F1": [r["test"]["macro_f1"] for r in runs],
            "test errors (of 10,000)": [float(r["test"]["errors"]) for r in runs],
            "val implied sum ceiling": [r["implied_sum_ceilings"]["val"]["exact_sum_accuracy"] for r in runs],
            "best epoch": [float(r["best"]["epoch"]) for r in runs],
        }
        for label, values in rows.items():
            print(f"  {label:<26} {mean_std(values)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
