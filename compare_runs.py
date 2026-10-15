"""Tabulate finished runs and check that they form a controlled comparison.

    python compare_runs.py runs/local-noaug runs/local-affine runs/local-elastic

A comparison is "controlled" only if every run used the same source pools and the same
validation/test manifests. This script checks that from summary.json and refuses to call the
table controlled otherwise. Single-seed differences can be noise; compare several train.seed values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    test_metrics = run_dir / "eval_test_best" / "metrics.json"
    summary["test_accuracy"] = json.loads(test_metrics.read_text())["main"]["accuracy"] if test_metrics.exists() else None
    return summary


def main(run_dirs: list[str]) -> int:
    if not run_dirs:
        print(__doc__)
        return 1
    runs = [load(Path(d)) for d in run_dirs]
    reference = runs[0]["data_ids"]
    shared_keys = ("val", "test", "val_text_only", "val_all_image", "test_text_only", "test_all_image")
    controlled = all(
        r["data_ids"]["source_pool_id"] == reference["source_pool_id"]
        and all(r["data_ids"]["manifest_ids"].get(k) == reference["manifest_ids"].get(k) for k in shared_keys)
        for r in runs
    )
    print(f"{'run':<34} {'augment':<15} {'seed':>4} {'eff.batch':>9} {'best step':>9} {'val acc':>8} {'val loss':>9} {'test acc':>8}")
    for r in runs:
        test = f"{r['test_accuracy']:.4f}" if r["test_accuracy"] is not None else "-"
        print(
            f"{Path(r['run_dir']).name:<34} {r['augment_mode']:<15} {r['seed']:>4} {r['effective_batch_size']:>9} "
            f"{r['best_step']:>9} {r['best_val_accuracy']:>8.4f} {r['best_val_loss']:>9.4f} {test:>8}"
        )
    same_train = len({r["data_ids"]["manifest_ids"]["train"] for r in runs}) == 1
    print(f"\nControlled comparison (same pools + eval manifests): {'YES' if controlled else 'NO'}")
    print(f"Same training manifest: {'yes' if same_train else 'no'}")
    return 0 if controlled else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
