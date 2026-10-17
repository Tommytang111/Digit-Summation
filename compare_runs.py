"""Tabulate finished runs and check that they form a controlled comparison.

    python compare_runs.py runs/local-noaug runs/local-affine runs/local-elastic
    python compare_runs.py runs/speed-w0 runs/speed-w2 runs/speed-w4      # num_workers speed study

A comparison is "controlled" only if every run used the same source pools and the same
validation/test manifests. This script checks that from summary.json and refuses to call the
table controlled otherwise. It also lists every config setting that differs between the runs, so a
speed comparison can confirm that only the loader settings changed. Single-seed accuracy
differences can be noise; compare several train.seed values.

Accuracy columns: val acc / val F1 / val loss describe the validation-selected checkpoint
(best.pt, chosen by accuracy); val F1 and test F1 are macro F1 over sum classes. Test columns stay
"-" until `evaluate.py --split test` has been run on best.pt.

Speed columns:
    seq/s       median of train/samples_per_sec over the run's log intervals (end to end: data
                loading + forward/backward/update; validation time excluded)
    train time  wall-clock seconds of the training session, including validation passes, worker
                startup, and checkpointing (for a resumed run: the last session only)
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

LOADER_KEYS = {"train.num_workers", "train.persistent_workers", "train.pin_memory"}
IGNORED_SECTIONS = {"logging", "resolved"}


def flatten(config: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def load(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    test_metrics = run_dir / "eval_test_best" / "metrics.json"
    test_main = json.loads(test_metrics.read_text())["main"] if test_metrics.exists() else {}
    summary["test_accuracy"] = test_main.get("accuracy")
    summary["test_macro_f1"] = test_main.get("macro_f1")  # absent for evaluations made before macro F1 existed
    records = [json.loads(line) for line in open(run_dir / "metrics.jsonl") if line.strip()]
    speeds = [r["train/samples_per_sec"] for r in records if "train/samples_per_sec" in r]
    summary["median_seq_per_sec"] = statistics.median(speeds) if speeds else None
    summary["config_flat"] = flatten({k: v for k, v in config.items() if k not in IGNORED_SECTIONS})
    return summary


def main(run_dirs: list[str]) -> int:
    if not run_dirs:
        print(__doc__)
        return 1
    runs = [load(Path(d)) for d in run_dirs]
    reference = runs[0]["data_ids"]
    # Every evaluation manifest (everything except training data) must be identical across runs.
    # This covers the main task (val, test, diagnostics) and the bonus (val_L01.., test_L01.., ...).
    shared_keys = sorted(k for k in reference["manifest_ids"] if not k.startswith("train"))
    controlled = all(
        r["data_ids"]["source_pool_id"] == reference["source_pool_id"]
        and all(r["data_ids"]["manifest_ids"].get(k) == reference["manifest_ids"].get(k) for k in shared_keys)
        for r in runs
    )

    fmt = lambda value: f"{value:.4f}" if value is not None else "-"  # noqa: E731
    print(
        f"{'run':<30} {'augment':<14} {'seed':>4} {'eff.batch':>9} {'best step':>9} {'val acc':>8} {'val F1':>7} "
        f"{'val loss':>9} {'test acc':>8} {'test F1':>7}"
    )
    for r in runs:
        print(
            f"{Path(r['run_dir']).name:<30} {r['augment_mode']:<14} {r['seed']:>4} {r['effective_batch_size']:>9} "
            f"{r['best_step']:>9} {r['best_val_accuracy']:>8.4f} {fmt(r.get('best_val_macro_f1')):>7} "
            f"{r['best_val_loss']:>9.4f} {fmt(r['test_accuracy']):>8} {fmt(r['test_macro_f1']):>7}"
        )

    print(f"\n{'run':<30} {'device':<7} {'workers':>7} {'persistent':>10} {'seq/s':>8} {'train time':>11} {'steps':>7}")
    for r in runs:
        flat = r["config_flat"]
        persistent = flat.get("train.persistent_workers", "-")  # absent in runs made before the option existed
        speed = f"{r['median_seq_per_sec']:.0f}" if r["median_seq_per_sec"] else "-"
        print(
            f"{Path(r['run_dir']).name:<30} {r['device']:<7} {flat['train.num_workers']:>7} {str(persistent):>10} "
            f"{speed:>8} {r['training_seconds_this_session']:>10.1f}s {r['final_optimizer_step']:>7}"
        )

    all_keys = set().union(*(r["config_flat"] for r in runs))
    differing = sorted(k for k in all_keys if len({json.dumps(r["config_flat"].get(k)) for r in runs}) > 1)
    print(f"\nControlled comparison (same pools + eval manifests): {'YES' if controlled else 'NO'}")
    train_ids = lambda r: tuple(sorted((k, v) for k, v in r["data_ids"]["manifest_ids"].items() if k.startswith("train")))  # noqa: E731
    print(f"Same training manifests: {'yes' if len({train_ids(r) for r in runs}) == 1 else 'no'}")
    print(f"Config settings that differ: {differing or 'none'}")
    other = [k for k in differing if k not in LOADER_KEYS and k != "name"]
    if other:
        print(f"  -> not a pure loader-speed comparison; non-loader settings also differ: {other}")
    return 0 if controlled else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
