"""Build (or verify) the frozen length-generalization manifests and run the leakage audit.

    python prepare_bonus_data.py --config configs/bonus_debug.yaml
    python prepare_bonus_data.py --config configs/bonus_local.yaml
    python prepare_bonus_data.py --config configs/bonus_local.yaml --audit-only

Writes bonus.manifest_dir/:
    source_pools.npz/json    the same 50k/10k/10k pools as the main task (at full fractions)
    manifests.json + *.npz   train_L01..L10, {val,test}_L01..L20, {val,test}_all_image_L01..L20
    leakage_audit.json       hard checks + all-text overlap report

Like prepare_data.py, re-running regenerates everything from the recorded seeds and checks it is
identical to what is on disk, refusing to overwrite unless --overwrite.
Exit codes: 0 = audit passed, 1 = audit failed, 2 = frozen-data mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from bonus_data import build_bonus_data, eval_lengths, manifest_name, train_lengths
from config import config_to_dict, load_config
from dataset import load_manifests, load_mnist_arrays, load_source_pools
from prepare_data import run_leakage_audit, save_or_verify_frozen

MAIN_FULL_POOLS = Path("data/manifests/full/source_pools.json")


def text_only_overlap_by_length(manifests: dict, cfg) -> list[dict]:
    """Per eval length: all-text rows in val/test whose exact input also occurs in training."""
    rows = []
    for length in eval_lengths(cfg):
        train_name = manifest_name("train", length)
        train_keys = set()
        if train_name in manifests:
            m = manifests[train_name]
            train_keys = {tuple(r) for r, img in zip(m.digits.tolist(), m.is_image.tolist()) if not any(img)}
        entry = {"group": length}
        for split in ("val", "test"):
            m = manifests[manifest_name(split, length)]
            text_rows = [tuple(r) for r, img in zip(m.digits.tolist(), m.is_image.tolist()) if not any(img)]
            entry[f"{split}_all_text_rows"] = len(text_rows)
            entry[f"{split}_all_text_seen_in_train"] = sum(r in train_keys for r in text_rows)
        rows.append(entry)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    directory = Path(cfg.bonus.manifest_dir)
    arrays = load_mnist_arrays(cfg.data.root, download=not args.no_download)
    if args.audit_only:
        pools, manifests = load_source_pools(directory), load_manifests(directory)
    else:
        pools, manifests = build_bonus_data(cfg, arrays)
        extra = {
            "config_name": cfg.name,
            "task": "bonus_decoder",
            "data_config": config_to_dict(cfg)["data"],
            "bonus_config": config_to_dict(cfg)["bonus"],
            "pool_id": pools.pool_id(),
            "created": datetime.now().isoformat(),
        }
        if not save_or_verify_frozen(directory, pools, manifests, extra, overwrite=args.overwrite):
            return 2

    d = cfg.data
    if (d.train_source_fraction, d.val_source_fraction, d.test_source_fraction) == (1.0, 1.0, 1.0) and MAIN_FULL_POOLS.exists():
        main_pool = json.loads(MAIN_FULL_POOLS.read_text())["pool_id"]
        same = main_pool == pools.pool_id()
        print(f"Image pools identical to the main task's full pools ({main_pool}): {'YES' if same else 'NO'}")

    report = run_leakage_audit(arrays, pools, manifests, text_only_duplicates="report")
    overlap = text_only_overlap_by_length(manifests, cfg)
    report["info"]["all_text_eval_rows_seen_in_training_by_length"] = overlap
    with open(directory / "leakage_audit.json", "w") as handle:
        json.dump(report, handle, indent=2)

    print("\n== Bonus manifests ==")
    print(f"pools {pools.report['final_counts']} (pool {pools.pool_id()})")
    train_rows = {L: manifests[manifest_name("train", L)].num_sequences for L in train_lengths(cfg)}
    print(f"train rows per length: {train_rows} (total {sum(train_rows.values()):,})")
    print(f"val/test rows per length: {cfg.bonus.num_eval_per_length} natural + {cfg.bonus.num_all_image_per_length} all-image, lengths 1-{cfg.bonus.max_eval_length}")
    print(f"\n{'L':>3} {'test sum range':>15} {'91-99':>7} {'>=100':>7} {'val all-text':>13} {'seen in train':>14} {'test all-text':>14} {'seen in train':>14}")
    for entry in overlap:
        m = manifests[manifest_name("test", entry["group"])]
        t = m.targets.astype(int)
        print(
            f"{entry['group']:>3} {f'{t.min()}-{t.max()}':>15} {int(((t >= 91) & (t <= 99)).sum()):>7} {int((t >= 100).sum()):>7} "
            f"{entry['val_all_text_rows']:>13} {entry['val_all_text_seen_in_train']:>14} {entry['test_all_text_rows']:>14} {entry['test_all_text_seen_in_train']:>14}"
        )
    print(f"\n== Leakage audit (text_only_duplicates=report): {report['num_checks'] - report['num_failed']}/{report['num_checks']} hard checks passed ==")
    for check in report["checks"]:
        if not check["passed"]:
            print(f"  FAIL  {check['check']}: {check['detail']}")
    shared = sum(item["shared_all_text_inputs"] for item in report["info"]["all_text_cross_manifest_overlap"])
    print(f"all-text inputs shared between same-length manifests (reported, not a failure): {shared}")
    print("AUDIT PASSED" if report["passed"] else "AUDIT FAILED")
    print(f"Full report: {directory / 'leakage_audit.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
