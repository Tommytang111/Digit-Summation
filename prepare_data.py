"""Download MNIST, isolate source pools, generate frozen manifests, and run the leakage audit.

    python prepare_data.py --config configs/debug.yaml               # build or verify, then audit
    python prepare_data.py --config configs/debug.yaml --audit-only  # re-audit what is on disk
    python prepare_data.py --config configs/local.yaml --overwrite   # deliberately replace frozen data

Manifests are frozen. Re-running regenerates everything deterministically from the recorded
seeds and checks that the result is identical to what is on disk. If the config changed the data
(counts, fractions, seeds), the script refuses to overwrite unless --overwrite is given.

Exit codes: 0 = audit passed, 1 = audit failed, 2 = frozen-manifest mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

from augmentations import build_transform
from config import AugmentConfig, config_to_dict, load_config
from dataset import (
    SPLIT_TO_PARTITION,
    SPLITS,
    DigitSumDataset,
    MnistArrays,
    SequenceManifest,
    SourcePools,
    build_source_pools,
    compute_targets,
    generate_manifest,
    input_keys,
    load_manifests,
    load_mnist_arrays,
    load_source_pools,
    save_manifests,
    save_source_pools,
)
from tokenizer import PAD_ID, encode_operand


def _diagnostic_seed(base: int, k: int) -> int:
    return int(np.random.SeedSequence([base, k]).generate_state(1)[0])


def manifest_plan(cfg) -> list[dict]:
    """Generation order matters: later manifests are resampled to avoid earlier ones' inputs.

    Test goes first so it is never altered, then validation, then training.
    """
    d = cfg.data
    n_diag = d.num_diagnostic_sequences
    return [
        {"name": "test", "pool_split": "test", "count": d.num_test_sequences, "p": d.image_probability, "seed": d.test_seed},
        {"name": "test_text_only", "pool_split": "test", "count": n_diag, "p": 0.0, "seed": _diagnostic_seed(d.diagnostic_seed, 0)},
        {"name": "test_all_image", "pool_split": "test", "count": n_diag, "p": 1.0, "seed": _diagnostic_seed(d.diagnostic_seed, 1)},
        {"name": "val", "pool_split": "val", "count": d.num_val_sequences, "p": d.image_probability, "seed": d.val_seed},
        {"name": "val_text_only", "pool_split": "val", "count": n_diag, "p": 0.0, "seed": _diagnostic_seed(d.diagnostic_seed, 2)},
        {"name": "val_all_image", "pool_split": "val", "count": n_diag, "p": 1.0, "seed": _diagnostic_seed(d.diagnostic_seed, 3)},
        {"name": "train", "pool_split": "train", "count": d.num_train_sequences, "p": d.image_probability, "seed": d.train_seed},
    ]


def build_all(cfg, arrays: MnistArrays) -> tuple[SourcePools, dict[str, SequenceManifest]]:
    d = cfg.data
    fractions = {"train": d.train_source_fraction, "val": d.val_source_fraction, "test": d.test_source_fraction}
    pools = build_source_pools(arrays, d.val_source_size, fractions, d.split_seed)
    manifests: dict[str, SequenceManifest] = {}
    seen_canonical: set[str] = set()
    for spec in manifest_plan(cfg):
        manifest = generate_manifest(
            name=spec["name"],
            pool_split=spec["pool_split"],
            pools=pools,
            arrays=arrays,
            num_sequences=spec["count"],
            sequence_length=d.sequence_length,
            image_probability=spec["p"],
            seed=spec["seed"],
            exclude_canonical=seen_canonical,
        )
        _, canonical = input_keys(manifest, arrays.hashes(manifest.partition))
        seen_canonical.update(canonical)
        manifests[spec["name"]] = manifest
    return pools, manifests


# ----------------------------------------------------------------------------- audit


def run_leakage_audit(arrays: MnistArrays, pools: SourcePools, manifests: dict[str, SequenceManifest]) -> dict:
    """Executable leakage audit. Hard checks decide `passed`; `info` is descriptive.

    Hard checks:
      - source pools are disjoint as (partition, index) IDs
      - every manifest's image operands come from its own split's pool; text slots have no source
      - no exact raw-pixel duplicate (SHA-256) is shared between pools
      - no exact complete-input duplicate and no permutation-equivalent input across manifests
      - targets equal text integers + MNIST labels, and image labels equal the sampled digits
      - model inputs carry no labels for images: image slots get PAD_ID tokens, text slots get
        the token of their digit string
    Not detectable here: near-duplicate images (e.g. a resaved or shifted copy) and same-writer
    images. MNIST provides no writer IDs, so validation is NOT writer-independent.
    """
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    namespaced = {split: {(SPLIT_TO_PARTITION[split], int(i)) for i in pools.ids[split]} for split in SPLITS}
    for a, b in combinations(SPLITS, 2):
        shared = namespaced[a] & namespaced[b]
        check(f"source IDs disjoint: {a} vs {b}", not shared, f"{len(shared)} shared (partition, index) IDs")

    pool_hashes = {split: set(arrays.hashes(SPLIT_TO_PARTITION[split])[pools.ids[split]].tolist()) for split in SPLITS}
    for a, b in combinations(SPLITS, 2):
        shared = pool_hashes[a] & pool_hashes[b]
        check(f"no identical raw pixels across pools: {a} vs {b}", not shared, f"{len(shared)} shared SHA-256 hashes")

    keys = {}
    info_manifests = {}
    for name, manifest in manifests.items():
        used = np.unique(manifest.source_index[manifest.is_image])
        outside = np.setdiff1d(used, pools.ids[manifest.pool_split])
        check(f"{name}: image IDs inside its {manifest.pool_split!r} pool", outside.size == 0, f"{outside.size} IDs outside")
        check(f"{name}: text slots have no image source", bool((manifest.source_index[~manifest.is_image] == -1).all()))

        labels = arrays.labels[manifest.partition]
        try:
            recomputed = compute_targets(manifest.digits, manifest.is_image, manifest.source_index, labels)
            consistent = bool(np.array_equal(recomputed, manifest.targets.astype(np.int64)))
            detail = "targets = text integers + MNIST labels"
        except AssertionError as error:
            consistent, detail = False, str(error)
        check(f"{name}: targets and image labels consistent", consistent, detail)

        keys[name] = input_keys(manifest, arrays.hashes(manifest.partition))
        exact, canonical = keys[name]
        image_slots = int(manifest.is_image.sum())
        info_manifests[name] = {
            "pool_split": manifest.pool_split,
            "sequences": manifest.num_sequences,
            "all_text_rows": int((~manifest.is_image).all(axis=1).sum()),
            "all_image_rows": int(manifest.is_image.all(axis=1).sum()),
            "within_split_exact_duplicate_rows": manifest.num_sequences - len(set(exact)),
            "within_split_permutation_duplicate_rows": manifest.num_sequences - len(set(canonical)),
            "image_slots": image_slots,
            "distinct_images_used": int(used.size),
            "mean_uses_per_distinct_image": round(image_slots / max(1, used.size), 3),
            "fraction_of_pool_used": round(used.size / max(1, len(pools.ids[manifest.pool_split])), 4),
            "distinct_target_sums": int(np.unique(manifest.targets).size),
        }

    names = list(manifests)
    for a, b in combinations(names, 2):
        exact_shared = set(keys[a][0]) & set(keys[b][0])
        canonical_shared = set(keys[a][1]) & set(keys[b][1])
        check(f"no exact complete-input duplicates: {a} vs {b}", not exact_shared, f"{len(exact_shared)} shared inputs")
        check(f"no permutation-equivalent inputs: {a} vs {b}", not canonical_shared, f"{len(canonical_shared)} shared multisets")

    # Inputs never encode image labels: build real dataset items (deterministic preprocessing).
    transform =build_transform(AugmentConfig(mode="none"), train=False)
    for name, manifest in manifests.items():
        dataset = DigitSumDataset(manifest, arrays.images[manifest.partition], transform)
        rows = range(min(200, len(dataset)))
        ok = True
        for row in rows:
            item = dataset[row]
            image_tokens = item["token_ids"][item["is_image"]]
            text_tokens = item["token_ids"][~item["is_image"]]
            expected_text = [encode_operand(int(d)) for d in item["digits"][~item["is_image"]]]
            ok &= bool((image_tokens == PAD_ID).all()) and text_tokens.tolist() == expected_text
        check(f"{name}: image slots carry PAD tokens only (first {len(rows)} rows)", ok)

    train_sums = set(np.unique(manifests["train"].targets).tolist()) if "train" in manifests else set()
    info = {
        "manifests": info_manifests,
        "legitimate_reuse": {
            "note": "Reused digit values and reused target sums across splits are expected and not leakage.",
            "test_sums_also_in_train": len(set(np.unique(manifests["test"].targets).tolist()) & train_sums) if "test" in manifests else None,
        },
        "limits": [
            "Exact hashes do not catch near-duplicate images (shifted, rescaled, re-encoded copies).",
            "MNIST ships no writer IDs: the train/val split is not writer-independent, and no writer-level claim is made.",
            "Permutation-equivalent rows are blocked across splits because this model has no positional "
            "embeddings and is therefore order-invariant.",
            "All-text rows depend only on digits; they are the only rows that could collide across splits "
            "and are resampled away during generation.",
        ],
    }
    passed = all(c["passed"] for c in checks)
    return {"passed": passed, "num_checks": len(checks), "num_failed": sum(not c["passed"] for c in checks), "checks": checks, "info": info}


def print_audit(report: dict, pools: SourcePools) -> None:
    r = pools.report
    print("\n== Source pools (exact-duplicate resolution) ==")
    print(f"official partitions: train {r['official_counts']['train']:,}, test {r['official_counts']['test']:,}")
    print(f"dev images identical to a test image (removed from dev): {r['dev_images_duplicating_test_removed']}")
    print(f"dev internal duplicate groups: {r['dev_internal_duplicate_groups']} (extra copies removed: {r['dev_internal_duplicates_removed']})")
    print(f"dev conflicting-label groups: {r['dev_conflicting_label_groups']} (images removed: {r['dev_conflicting_label_images_removed']})")
    print(f"test internal duplicate groups (kept): {r['test_internal_duplicate_groups_kept']}")
    print(f"full pools: train {r['full_train_pool']:,} (shortfall vs 50,000: {r['train_shortfall_vs_50000']}), val {r['full_val_pool']:,}")
    print(f"after source fractions {r['fractions']}: {r['final_counts']}")
    print("\n== Manifests ==")
    for name, info in report["info"]["manifests"].items():
        print(
            f"{name:>15}: {info['sequences']:>6,} seq | all-text {info['all_text_rows']:>4} | all-image {info['all_image_rows']:>4} | "
            f"distinct images {info['distinct_images_used']:>6,} ({info['mean_uses_per_distinct_image']:.2f} uses each) | "
            f"within-split perm-dups {info['within_split_permutation_duplicate_rows']}"
        )
    print(f"\n== Leakage audit: {report['num_checks'] - report['num_failed']}/{report['num_checks']} hard checks passed ==")
    for c in report["checks"]:
        if not c["passed"]:
            print(f"  FAIL  {c['check']}: {c['detail']}")
    print("AUDIT PASSED" if report["passed"] else "AUDIT FAILED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--overwrite", action="store_true", help="replace frozen pools/manifests if they differ")
    parser.add_argument("--audit-only", action="store_true", help="audit what is on disk without regenerating")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    directory = Path(cfg.data.manifest_dir)
    arrays = load_mnist_arrays(cfg.data.root, download=not args.no_download)

    if args.audit_only:
        pools, manifests = load_source_pools(directory), load_manifests(directory)
    else:
        pools, manifests = build_all(cfg, arrays)
        if (directory / "manifests.json").exists() and not args.overwrite:
            disk_pools, disk_manifests = load_source_pools(directory), load_manifests(directory)
            differences = []
            if disk_pools.pool_id() != pools.pool_id():
                differences.append("source pools")
            for name in sorted(set(manifests) | set(disk_manifests)):
                if name not in manifests or name not in disk_manifests or manifests[name].manifest_id() != disk_manifests[name].manifest_id():
                    differences.append(name)
            if differences:
                print(f"Frozen data in {directory} differs from what this config generates: {differences}")
                print("Refusing to overwrite. Use a different data.manifest_dir, or pass --overwrite deliberately.")
                return 2
            print(f"Frozen data in {directory} verified: regeneration is identical (pool {pools.pool_id()}). Nothing rewritten.")
        else:
            save_source_pools(pools, directory)
            extra = {"config_name": cfg.name, "data_config": config_to_dict(cfg)["data"], "pool_id": pools.pool_id(), "created": datetime.now().isoformat()}
            save_manifests(manifests, directory, extra)
            print(f"Wrote source pools + {len(manifests)} manifests to {directory}")

    report = run_leakage_audit(arrays, pools, manifests)
    with open(directory / "leakage_audit.json", "w") as handle:
        json.dump(report, handle, indent=2)
    print_audit(report, pools)
    print(f"Full report: {directory / 'leakage_audit.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
