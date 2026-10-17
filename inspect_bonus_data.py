"""Inspect the length-generalization data before training.

    python inspect_bonus_data.py --config configs/bonus_debug.yaml      # writes runs/inspect-<name>/

Prints:
  - answer tokenization and the teacher-forcing shift for example rows at several lengths;
  - the batch tensor contract after bonus_collate (variable lengths, right padding, masks);
  - per-length statistics: rows, mean image operands, sum range, share of sums 91-99 and >= 100.
Writes examples_by_length.png and large_sums_by_length.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import visualization as viz
from bonus_data import bonus_collate, eval_lengths, manifest_name, train_lengths
from bonus_engine import evaluation_dataset, load_bonus_data
from config import load_config
from tokenizer import OUTPUT_ID_TO_TOKEN, answer_ids, answer_text, number_to_string, teacher_forcing_pair


def tokens(ids) -> str:
    return "[" + ", ".join(OUTPUT_ID_TO_TOKEN[int(i)] for i in ids) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    data = load_bonus_data(cfg)
    out = Path(f"runs/inspect-{cfg.name}")
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"config": cfg.name}

    shown_lengths = sorted({1, 3, cfg.bonus.max_train_length, cfg.bonus.max_train_length + 1, cfg.bonus.max_eval_length})
    print("== Answers as characters: target sum -> canonical text -> output tokens -> teacher forcing ==")
    examples = []
    for L in shown_lengths:
        m = data.manifests[manifest_name("val", L)]
        row = int(np.argmax(m.targets)) if L == cfg.bonus.max_eval_length else 0  # show the largest sum at the longest length
        operands = " ".join(f"[img {int(d)}]" if img else number_to_string(int(d)) for d, img in zip(m.digits[row], m.is_image[row]))
        text = answer_text(int(m.targets[row]))
        ids = answer_ids(text)
        decoder_input, decoder_target = teacher_forcing_pair(ids)
        print(f"L={L:>2}: {operands}\n      sum {int(m.targets[row])} -> {text!r} -> {tokens(ids)} = {ids}\n"
              f"      decoder input {tokens(decoder_input)}  ->  targets {tokens(decoder_target)}")
        examples.append({"length": L, "operands": operands, "answer": text, "ids": ids})
    summary["answer_examples"] = examples

    dataset = evaluation_dataset(cfg, data, [manifest_name("val", L) for L in (1, 4, cfg.bonus.max_train_length)])
    items = [dataset[0], dataset[len(dataset) // 2], dataset[len(dataset) - 1]]
    batch = bonus_collate(items)
    print("\n== Batch contract (rows of lengths " + ", ".join(str(int(n)) for n in batch["lengths"]) + "; right-padded) ==")
    for key, value in batch.items():
        print(f"  {key:<21} {str(value.dtype):<14} {list(value.shape)}")
    print("  padding_mask (True = padding):\n" + "\n".join(f"    {row}" for row in batch["padding_mask"].int().tolist()))
    print("  decoder_input / decoder_target (output IDs; 0 = PAD, 1 = BOS, 2 = EOS, digit d = d + 3):")
    for i, t in zip(batch["decoder_input"].tolist(), batch["decoder_target"].tolist()):
        print(f"    {i}  ->  {t}")

    print(f"\n== Per-length statistics (test; training lengths {train_lengths(cfg)[0]}-{train_lengths(cfg)[-1]}, sums in training <= {9 * cfg.bonus.max_train_length}) ==")
    print(f"{'L':>3} {'rows':>6} {'mean images':>12} {'sum min-max':>12} {'91-99':>8} {'>=100':>8}")
    stats = []
    for L in eval_lengths(cfg):
        m = data.manifests[manifest_name("test", L)]
        t = m.targets.astype(int)
        entry = {"group": L, "rows": int(len(t)), "mean_images": float(m.is_image.sum(1).mean()), "min": int(t.min()), "max": int(t.max()),
                 "share_91_99": float(((t >= 91) & (t <= 99)).mean()), "share_ge_100": float((t >= 100).mean())}
        stats.append(entry)
        print(f"{L:>3} {entry['rows']:>6} {entry['mean_images']:>12.2f} {f'{entry['min']}-{entry['max']}':>12} {entry['share_91_99']:>8.3f} {entry['share_ge_100']:>8.3f}")
    summary["test_length_statistics"] = stats
    viz.plot_by_length(
        {"share of sums 91-99": [(s["group"], s["share_91_99"]) for s in stats], "share of sums >= 100 (three-digit answers)": [(s["group"], s["share_ge_100"]) for s in stats]},
        out / "large_sums_by_length.png", "Test sequences: answers beyond the training range, by length", "Fraction of rows",
        cfg.bonus.max_train_length, {s["group"]: s["rows"] for s in stats}, ylim=None,
    )
    rows = []
    for L in shown_lengths:
        m = data.manifests[manifest_name("val", L)]
        rows += viz.manifest_gallery_rows(m, data.arrays.images[m.partition], [0], captions=[f"L={L}"])
    viz.plot_sequence_gallery(rows, out / "examples_by_length.png", "Validation rows at several lengths (raw pixels; text operands as strings)")
    (out / "inspection_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
