"""Incremental checks for the length-generalization bonus (run before any real training).

    python bonus_checks.py shapes    --config configs/bonus_debug.yaml   # every tensor boundary, all variants
    python bonus_checks.py step      --config configs/bonus_debug.yaml   # one real teacher-forced update
    python bonus_checks.py overfit   --config configs/bonus_debug.yaml   # memorize 64 fixed rows, <= 300 updates
    python bonus_checks.py structure --config configs/bonus_debug.yaml   # invariances that need no training

Add `--set bonus.architecture=items_only_rope` (or items_only_alibi) to check another variant.
`structure` covers all five variants unless --checkpoint is given.

`structure` reports properties of the architecture itself, not learned skills:
  - padding: extra right padding must not change outputs (any correct implementation);
  - items_only_no_positions: identical repeated operands [3], [3,3], [3,3,3], ... must give
    identical next-token distributions, because softmax attention averages identical items
    (count-blind by construction; more training cannot fix it);
  - sinusoidal variants: the same operand gets a different representation at each position.
    That makes counting *possible*, but does not make it learned.
  - RoPE/ALiBi items-only variants: relative attention still averages identical value vectors,
    so perfectly identical repeated operands remain count-blind (a documented limitation).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import visualization as viz
from bonus_data import bonus_collate
from bonus_engine import load_bonus_data, load_bonus_model, score, summarize, training_dataset
from bonus_model import ARCHITECTURES, bonus_model_inputs, build_bonus_model, position_schemes, teacher_forced_loss
from config import load_config
from engine import build_optimizer, make_loader, resolve_device, seed_everything
from model import count_parameters
from tokenizer import BOS_ID, OUT_PAD_ID, PAD_ID, encode_operand


def text_batch(rows: list[list[int]], device) -> dict:
    """Right-padded all-text batch from digit rows (no data needed). Digit d -> token encode_operand(d)."""
    length = max(len(r) for r in rows)
    token_ids = torch.full((len(rows), length), PAD_ID, dtype=torch.long)
    padding_mask = torch.ones(len(rows), length, dtype=torch.bool)
    for i, row in enumerate(rows):
        token_ids[i, : len(row)] = torch.tensor([encode_operand(d) for d in row])
        padding_mask[i, : len(row)] = False
    return {
        "images": torch.zeros(len(rows), length, 1, 28, 28, device=device),
        "token_ids": token_ids.to(device),
        "is_image": torch.zeros(len(rows), length, dtype=torch.bool, device=device),
        "padding_mask": padding_mask.to(device),
    }


def with_extra_padding(inputs: dict, extra: int) -> dict:
    """Append `extra` padded slots holding garbage (random pixels, real-looking tokens, is_image True)."""
    batch, _ = inputs["token_ids"].shape
    device = inputs["token_ids"].device
    return {
        "images": torch.cat([inputs["images"], torch.randn(batch, extra, 1, 28, 28, device=device) * 5], dim=1),
        "token_ids": torch.cat([inputs["token_ids"], torch.randint(1, 11, (batch, extra), device=device)], dim=1),
        "is_image": torch.cat([inputs["is_image"], torch.ones(batch, extra, dtype=torch.bool, device=device)], dim=1),
        "padding_mask": torch.cat([inputs["padding_mask"], torch.ones(batch, extra, dtype=torch.bool, device=device)], dim=1),
    }


@torch.no_grad()
def next_token_distribution(model, inputs: dict, prefix: list[int]) -> torch.Tensor:
    """Softmax over the next answer token after `prefix` (starting with BOS): [B, 13]."""
    memory = model.encode(**inputs)
    decoder_input = torch.tensor([prefix] * memory.shape[0], device=memory.device)
    return model.decode(memory, inputs["padding_mask"], decoder_input)[:, -1].softmax(dim=-1)


@torch.no_grad()
def structural_report(model, device, max_length: int = 20, real_inputs: dict | None = None) -> dict:
    """Architecture properties (eval mode). Values are max absolute differences."""
    model.eval()
    report = {"architecture": model.architecture}
    # 1. Right padding must not change anything.
    inputs = real_inputs or text_batch([[3], [4, 0, 5], [9, 9, 1, 0, 2, 7, 3], [0] * 12], device)
    memory = model.encode(**inputs)
    padded = with_extra_padding(inputs, extra=5)
    decoder_input = torch.full((memory.shape[0], 1), BOS_ID, device=device)
    logits = model.decode(memory, inputs["padding_mask"], decoder_input)
    logits_padded = model.decode(model.encode(**padded), padded["padding_mask"], decoder_input)
    report["padding_max_abs_logit_diff"] = float((logits - logits_padded).abs().max())
    report["padding_generation_identical"] = bool(torch.equal(model.generate(**inputs), model.generate(**padded)))
    # 2. Repeated identical text operands [3]*k, k = 1..max_length, same decoder prefix.
    for prefix_name, prefix in (("BOS", [BOS_ID]), ("BOS+'3'", [BOS_ID, 6])):
        dists = [next_token_distribution(model, text_batch([[3] * k], device), prefix) for k in range(1, max_length + 1)]
        report[f"repeated_3_max_dist_diff_vs_k1[{prefix_name}]"] = float(max((d - dists[0]).abs().max() for d in dists))
    # 3. Do identical operands get different representations at different positions?
    memory = model.encode(**text_batch([[3] * 5], device))[0]  # [5, D]
    report["identical_operands_max_diff_across_positions"] = float((memory - memory[0]).abs().max())
    return report


# ----------------------------------------------------------------------------- commands


def command_shapes(args) -> int:
    cfg = load_config(args.config, args.overrides)
    device = resolve_device(args.device)
    rows = [[3], [4, 0, 5, 9, 1], [9] * 10, [1] * 20]  # lengths 1, 5, 10, 20 -> right-padded to 20
    for architecture in ARCHITECTURES:
        cfg.bonus.architecture = architecture
        torch.manual_seed(0)
        model = build_bonus_model(cfg).to(device).eval()
        inputs = text_batch(rows, device)
        decoder_input = torch.tensor([[BOS_ID, 6, OUT_PAD_ID], [BOS_ID, 4, 3], [BOS_ID, 12, 3], [BOS_ID, 5, 3]], device=device)
        decoder_padding = decoder_input == OUT_PAD_ID
        with torch.no_grad():
            features = model.operands.embed_operands(**inputs)
            memory = model.encode(**inputs)
            logits = model.decode(memory, inputs["padding_mask"], decoder_input, decoder_padding)
            generated = model.generate(**inputs, max_new_tokens=cfg.bonus.max_new_tokens)
        print(f"\n== {architecture} ({count_parameters(model)['total']:,} parameters) ==")
        print(f"operands: token_ids {list(inputs['token_ids'].shape)} padding_mask {list(inputs['padding_mask'].shape)} (lengths 1/5/10/20)")
        schemes = position_schemes(architecture)
        print(f"  embed_operands -> {list(features.shape)}; operand position scheme: {schemes['operands']}")
        print(f"  encoder memory -> {list(memory.shape)}")
        print(f"decoder_input {list(decoder_input.shape)} (answer position scheme: {schemes['answer']}), causal mask [3, 3], padding mask {list(decoder_padding.shape)}")
        print(f"  logits -> {list(logits.shape)} (output vocabulary 13); summary path: {model.summary_projection is not None}")
        print(f"  generate (greedy, cap {cfg.bonus.max_new_tokens}) -> {list(generated.shape)}")
    return 0


def command_step(args) -> int:
    cfg = load_config(args.config, args.overrides)
    device = resolve_device(args.device)
    seed_everything(cfg.train.seed)
    data = load_bonus_data(cfg)
    dataset = training_dataset(cfg, data)
    rows = np.random.default_rng(0).choice(len(dataset), size=cfg.train.batch_size, replace=False)
    batch = bonus_collate([dataset[(int(r), 0)] for r in rows])
    model = build_bonus_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    inputs, targets = bonus_model_inputs(batch, device), batch["decoder_target"].to(device)
    print(f"{cfg.bonus.architecture} on {device}: lengths {sorted(set(batch['lengths'].tolist()))}, answers up to {int(batch['targets'].max())}")
    model.eval()
    with torch.no_grad():
        before = float(teacher_forced_loss(model(**inputs), targets))
    model.train()
    loss = teacher_forced_loss(model(**inputs), targets)
    loss.backward()
    ok = True
    print(f"initial teacher-forced loss {before:.4f} (uniform over 13 tokens = ln 13 = {np.log(13):.4f})")
    groups = {
        "operands.image_encoder": model.operands.image_encoder,
        "operands.token_embedding": model.operands.token_embedding,
        "operands.layers": model.operands.layers,
        "answer_embedding": model.answer_embedding,
        "decoder_layers": model.decoder_layers,
        "vocab_head": model.vocab_head,
    }
    if model.summary_projection is not None:
        groups["summary_projection"] = model.summary_projection
    for name, module in groups.items():
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        norm = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads))) if grads else 0.0
        finite = all(bool(torch.isfinite(g).all()) for g in grads)
        ok &= finite and norm > 0
        print(f"  {name:<26} grad norm {norm:.4e}  finite={finite}")
    pad_grad = float(model.answer_embedding.weight.grad[OUT_PAD_ID].abs().max())
    print(f"  output-PAD embedding row gradient (must be 0): {pad_grad}")
    ok &= pad_grad == 0.0
    optimizer.step()
    model.eval()
    with torch.no_grad():
        after = float(teacher_forced_loss(model(**inputs), targets))
    print(f"teacher-forced loss on this batch {before:.4f} -> {after:.4f} after one AdamW update (wiring check only)")
    print("STEP CHECK PASSED" if ok else "STEP CHECK FAILED")
    return 0 if ok else 1


def command_overfit(args) -> int:
    cfg = load_config(args.config, args.overrides)
    o = cfg.overfit
    max_steps = args.max_steps or o.max_steps
    device = resolve_device(args.device)
    seed_everything(o.seed)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, dropout=0.0), bonus=dataclasses.replace(cfg.bonus, decoder_dropout=0.0))
    data = load_bonus_data(cfg)
    full = training_dataset(cfg, data, augment=False)  # augmentation off
    rows = np.sort(np.random.default_rng(o.seed).choice(len(full), size=o.num_examples, replace=False))
    subset = torch.utils.data.Subset(full, rows.tolist())  # the same fixed rows every step (int keys: no augmentation reseed)
    loader = make_loader(subset, o.batch_size, device, cfg.train, collate_fn=bonus_collate)
    eval_loader = make_loader(subset, 256, device, cfg.train, collate_fn=bonus_collate)
    model = build_bonus_model(cfg).to(device)
    optimizer = build_optimizer(model, o.lr, weight_decay=0.0)
    lengths = sorted(set(full.row_lengths[rows].tolist()))
    print(f"Overfit {cfg.bonus.architecture}: {o.num_examples} fixed training rows (lengths {lengths}), aug off, dropout 0, lr {o.lr}, <= {max_steps} updates, {device}")
    history, step, reached = [], 0, None
    while step < max_steps and reached is None:
        for batch in loader:
            model.train()
            loss = teacher_forced_loss(model(**bonus_model_inputs(batch, device)), batch["decoder_target"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step % o.eval_every_steps == 0 or step == max_steps:
                result = score(model, eval_loader, device, cfg)
                summary = summarize(result)
                history.append({"optimizer_step": step, "loss": result["tf_loss"], "accuracy": summary["accuracy"], "invalid_rate": summary["invalid_rate"]})
                print(f"  step {step:>4}  teacher-forced loss {result['tf_loss']:.4f}  free-generation exact accuracy {summary['accuracy']:.3f}  invalid {summary['invalid_rate']:.3f}")
                if summary["accuracy"] >= o.target_accuracy:
                    reached = step
                    break
            if step >= max_steps:
                break
    out = Path(f"runs/bonus-overfit-{cfg.bonus.architecture}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    outcome = {"success": reached is not None, "reached_at_step": reached, "max_steps": max_steps, "final": history[-1] if history else None,
               "best_accuracy": max((h["accuracy"] for h in history), default=0.0), "architecture": cfg.bonus.architecture, "lengths": lengths}
    if reached is None and history:
        final = score(model, eval_loader, device, cfg)
        wrong = np.flatnonzero(~final["correct"])
        outcome["remaining_errors"] = [
            {"length": int(final["length"][i]), "target": int(final["target"][i]), "generated": final["text"][i], "status": str(final["status"][i])} for i in wrong[:20]
        ]
    (out / "overfit_result.json").write_text(json.dumps({"outcome": outcome, "history": history}, indent=2))
    viz.plot_overfit_curve(history, out / "overfit_curve.png", f"Bonus tiny-overfit ({cfg.bonus.architecture}): {o.num_examples} fixed rows, free generation")
    if reached is not None:
        print(f"OVERFIT CHECK PASSED at step {reached}: every fixed row's full answer is generated exactly (memorization only, not generalization).")
    else:
        print(f"OVERFIT CHECK NOT PASSED within {max_steps} updates (best exact accuracy {outcome['best_accuracy']:.3f}); see {out}/overfit_result.json")
    print(f"Saved {out}")
    return 0 if reached is not None else 1


def command_structure(args) -> int:
    device = resolve_device(args.device)
    if args.checkpoint:
        model, _, _ = load_bonus_model(Path(args.checkpoint), device)
        reports = [structural_report(model, device)]
    else:
        cfg = load_config(args.config, args.overrides)
        reports = []
        for architecture in ARCHITECTURES:
            cfg.bonus.architecture = architecture
            torch.manual_seed(0)
            reports.append(structural_report(build_bonus_model(cfg).to(device), device))
    print("Structural properties (untrained unless --checkpoint; eval mode; max absolute differences):")
    for report in reports:
        print(f"\n  {report['architecture']}")
        for key, value in report.items():
            if key != "architecture":
                print(f"    {key:<55} {value:.2e}" if isinstance(value, float) else f"    {key:<55} {value}")
    print("\nPadding diffs must be ~0 for every variant. No-position, RoPE, and ALiBi items-only variants have ~0 repeated-operand differences")
    print("(count-blind by construction). Sinusoidal variants differ across positions, which makes counting possible, not learned.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("shapes", "step", "overfit", "structure"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/bonus_debug.yaml")
        p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
        p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
        if name == "overfit":
            p.add_argument("--max-steps", type=int, default=None, help="override overfit.max_steps")
        if name == "structure":
            p.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    return {"shapes": command_shapes, "step": command_step, "overfit": command_overfit, "structure": command_structure}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
