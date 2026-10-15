"""Incremental diagnostics, run in this order before any real training:

    python sanity_checks.py env                               # versions, devices, MPS op probe
    python sanity_checks.py shapes                            # shape walk through every model boundary
    python sanity_checks.py step   --config configs/debug.yaml   # one real forward/backward/update
    python sanity_checks.py overfit --config configs/debug.yaml  # memorize N fixed examples (<= max_steps)

`overfit` forces augmentation=none and dropout=0 and trains on the same fixed examples. If the
wiring, labels, and gradients are right, it should reach overfit.target_accuracy on those examples
(measured in eval mode). It exits 0 on success and 1 otherwise. Failure means: investigate
before scaling up. Success only shows that the model can memorize; it says nothing about
generalization.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import visualization as viz
from augmentations import build_transform
from config import ExperimentConfig, load_config
from dataset import collate_batch, generate_manifest, load_prepared_data
from engine import (
    build_optimizer,
    device_report,
    evaluate,
    make_loader,
    mps_fallback_enabled,
    resolve_device,
    seed_everything,
    synchronize,
    warn_if_mps_fallback,
)
from model import build_model, count_parameters, model_inputs
from tokenizer import PAD_ID


def synthetic_batch(batch_size: int = 4, length: int = 10, device="cpu", kind: str = "mixed", pad_last: int = 0, seed: int = 0) -> dict:
    """Random batch obeying the tensor contract; `pad_last` pads the last row's final slots."""
    generator = torch.Generator().manual_seed(seed)
    digits = torch.randint(0, 10, (batch_size, length), generator=generator)
    if kind == "mixed":
        is_image = torch.rand(batch_size, length, generator=generator) < 0.5
    else:
        is_image = torch.full((batch_size, length), kind == "all_image")
    padding_mask = torch.zeros(batch_size, length, dtype=torch.bool)
    if pad_last:
        padding_mask[-1, length - pad_last :] = True
    is_image &= ~padding_mask
    token_ids = (digits + 1).masked_fill(is_image | padding_mask, PAD_ID)
    images = torch.randn(batch_size, length, 1, 28, 28, generator=generator) * is_image[..., None, None, None]
    return {
        "images": images.to(device),
        "token_ids": token_ids.to(device),
        "is_image": is_image.to(device),
        "padding_mask": padding_mask.to(device),
        "targets": digits.masked_fill(padding_mask, 0).sum(dim=1).to(device),
    }


def command_env(_args) -> int:
    print("== Versions ==")
    for name in ("torch", "torchvision", "numpy", "albumentations", "albucore", "cv2", "matplotlib", "yaml", "wandb", "pytest"):
        try:
            module = importlib.import_module(name)
            print(f"{name:>15}: {getattr(module, '__version__', '?')}")
        except ImportError:
            print(f"{name:>15}: NOT INSTALLED")
    print(f"{'python':>15}: {sys.version.split()[0]}")
    print("\n== Devices ==")
    print(f"MPS built={torch.backends.mps.is_built()} available={torch.backends.mps.is_available()}")
    print(f"CUDA available={torch.cuda.is_available()}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print(f"PYTORCH_ENABLE_MPS_FALLBACK set: {mps_fallback_enabled()}")
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])
    print("\n== Op probe: forward + backward + AdamW step for mixed / all-text / all-image / padded batches ==")
    failures = 0
    for name in devices:
        device = torch.device(name)
        warn_if_mps_fallback(device)
        try:
            torch.manual_seed(0)
            model = build_model(ExperimentConfig()).to(device)
            optimizer = build_optimizer(model, 3e-4, 0.01)
            for kind, pad in (("mixed", 0), ("all_text", 0), ("all_image", 0), ("mixed", 3)):
                batch = synthetic_batch(device=device, kind=kind, pad_last=pad)
                logits = model(**{k: batch[k] for k in ("images", "token_ids", "is_image", "padding_mask")})
                loss = F.cross_entropy(logits, batch["targets"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss for {kind}")
            synchronize(device)
            print(f"  {name}: OK  ({device_report(device)})")
        except Exception as error:  # report every device, then fail
            failures += 1
            print(f"  {name}: FAILED: {type(error).__name__}: {error}")
    return 1 if failures else 0


def command_shapes(args) -> int:
    cfg = load_config(args.config) if args.config else ExperimentConfig()
    device = resolve_device(args.device)
    torch.manual_seed(0)
    model = build_model(cfg).to(device).eval()
    batch = synthetic_batch(batch_size=4, length=cfg.data.sequence_length, device=device, pad_last=2)
    inputs = {k: batch[k] for k in ("images", "token_ids", "is_image", "padding_mask")}
    print(f"device {device}; parameters: {count_parameters(model)}")
    print("\n== Inputs (batch of 4; last row padded by 2 to exercise masks) ==")
    for key, value in inputs.items():
        print(f"{key:>13}: {str(value.dtype):<14} {list(value.shape)}")
    with torch.no_grad():
        image_slots = inputs["is_image"] & ~inputs["padding_mask"]
        selected = inputs["images"][image_slots]
        print("\n== Image encoder ==")
        print(f"selected image slots       {list(selected.shape)}   (N_img = {int(image_slots.sum())})")
        x = selected
        for layer in model.image_encoder.features:
            x = layer(x)
            print(f"  after {layer.__class__.__name__:<10} {list(x.shape)}")
        print(f"  flatten + Linear + LayerNorm -> {list(model.image_encoder(selected).shape)}")
        print("\n== Token embedding ==")
        print(f"token_ids {list(inputs['token_ids'].shape)} -> {list(model.token_embedding(inputs['token_ids']).shape)}")
        print(f"PAD row of the embedding is all zeros: {bool((model.token_embedding.weight[PAD_ID] == 0).all())}")
        features = model.embed_operands(**inputs)
        print(f"\nselected per-slot features (+ modality embedding) {list(features.shape)}")
        hidden = model.encode(features, inputs["padding_mask"])
        print(f"Transformer encoder ({len(model.layers)} independent layers, nhead={cfg.model.nhead}, "
              f"head dim={cfg.model.d_model // cfg.model.nhead}, ff={cfg.model.dim_feedforward}) -> {list(hidden.shape)}")
        pooled = model.pool(hidden, inputs["padding_mask"])
        print(f"masked sum pooling -> {list(pooled.shape)}")
        logits = model.head(pooled)
        print(f"linear head -> logits {list(logits.shape)}")
        direct = model(**inputs)
        print(f"\nstep-by-step logits == model.forward logits: {torch.allclose(logits, direct, atol=1e-5)}")
        same_init = torch.equal(model.layers[0].linear1.weight, model.layers[1].linear1.weight)
        print(f"encoder layers independently initialized (weights differ): {not same_init}")
    return 0


def command_step(args) -> int:
    cfg = load_config(args.config, args.overrides)
    device = resolve_device(args.device)
    seed_everything(cfg.train.seed)
    data = load_prepared_data(cfg.data)
    dataset = data.dataset("train", build_transform(cfg.augment, train=True, seed=1))
    batch = collate_batch([dataset[i] for i in range(cfg.train.batch_size)])
    inputs, targets = model_inputs(batch, device), batch["targets"].to(device)
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)

    model.eval()  # measure the same batch without dropout before/after the updates
    with torch.no_grad():
        loss_before = F.cross_entropy(model(**inputs), targets).item()
    print(f"device {device}; batch: {int(batch['is_image'].sum())} image slots, {int((~batch['is_image']).sum())} text slots")
    print(f"initial loss {loss_before:.4f} (uniform prediction over {cfg.num_classes} classes = ln {cfg.num_classes} = {np.log(cfg.num_classes):.4f})")

    def update(label: str, require_all: bool) -> bool:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(**inputs), targets).backward()
        print(f"\n== {label}: gradient norm per component ==")
        ok = True
        for name, module in model.named_children():
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            norm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads)).item() if grads else 0.0
            finite = all(bool(torch.isfinite(g).all()) for g in grads)
            ok &= finite and (norm > 0 or not require_all)
            print(f"  {name:<20} grad norm {norm:.4e}  finite={finite}")
        pad_grad = model.token_embedding.weight.grad[PAD_ID].abs().max().item()
        print(f"  PAD embedding row gradient (must be 0): {pad_grad}")
        ok &= pad_grad == 0.0
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm or float("inf"))
        optimizer.step()
        print(f"  pre-clip total grad norm {grad_norm.item():.3f}")
        return ok

    # The head is zero-initialized, so on the first update the backbone receives exactly zero
    # gradient (dL/dpooled = W^T (p - y) = 0). From the second update on, every component must.
    first_ok = update("Update 1 (only the zero-initialized head is expected to receive gradient)", require_all=False)
    second_ok = update("Update 2 (every component must receive finite, non-zero gradient)", require_all=True)
    model.eval()
    with torch.no_grad():
        loss_after = F.cross_entropy(model(**inputs), targets).item()
    print(f"\nloss on this batch: {loss_before:.4f} -> {loss_after:.4f} after two AdamW updates")
    print("(two updates on one batch are a wiring check, not evidence of learning)")
    ok = first_ok and second_ok
    print("STEP CHECK PASSED" if ok else "STEP CHECK FAILED")
    return 0 if ok else 1


def command_overfit(args) -> int:
    cfg = load_config(args.config, args.overrides)
    o = cfg.overfit
    max_steps = args.max_steps or o.max_steps
    device = resolve_device(args.device)
    seed_everything(o.seed)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, dropout=0.0))  # dropout off
    data = load_prepared_data(cfg.data)
    if o.image_probability is None:
        rows = np.sort(np.random.default_rng(o.seed).choice(data.manifests["train"].num_sequences, size=o.num_examples, replace=False))
        fixed = data.manifests["train"].subset(rows, name="overfit")
    else:  # fresh rows drawn from the training pool only
        fixed = generate_manifest(
            "overfit", "train", data.pools, data.arrays, o.num_examples, cfg.data.sequence_length, o.image_probability, seed=o.seed
        )
    print(f"fixed set: {fixed.num_sequences} rows, {int(fixed.is_image.sum())} image / {int((~fixed.is_image).sum())} text operands")
    dataset = data.dataset(fixed, build_transform(cfg.augment, train=False))  # no augmentation
    loader = make_loader(dataset, o.batch_size, device, cfg.train)
    eval_loader = make_loader(dataset, 256, device, cfg.train)

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, o.lr, weight_decay=0.0)
    print(
        f"Overfit check: {o.num_examples} fixed training rows, batch {o.batch_size}, lr {o.lr}, "
        f"dropout 0, augmentation none, up to {max_steps} optimizer steps, device {device}"
    )
    history, step, reached_at = [], 0, None
    model.train()
    while step < max_steps and reached_at is None:
        for batch in loader:
            loss = F.cross_entropy(model(**model_inputs(batch, device)), batch["targets"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step % o.eval_every_steps == 0 or step == max_steps:
                result = evaluate(model, eval_loader, device)
                history.append({"optimizer_step": step, **result.metrics()})
                print(f"  step {step:>4}  fixed-set loss {result.loss:.4f}  accuracy {result.accuracy:.3f}")
                if result.accuracy >= o.target_accuracy:
                    reached_at = step
                    break
            if step >= max_steps:
                break

    out = Path(f"runs/overfit-{cfg.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    outcome = {
        "success": reached_at is not None,
        "reached_at_step": reached_at,
        "max_steps": max_steps,
        "target_accuracy": o.target_accuracy,
        "final": history[-1] if history else None,
        "best_accuracy": max((h["accuracy"] for h in history), default=0.0),
        "settings": {**dataclasses.asdict(o), "max_steps": max_steps, "device": str(device)},
    }
    with open(out / "overfit_result.json", "w") as handle:
        json.dump({"outcome": outcome, "history": history}, handle, indent=2)
    viz.plot_overfit_curve(history, out / "overfit_curve.png", f"Tiny-overfit check: {o.num_examples} fixed examples")
    if outcome["success"]:
        print(f"OVERFIT CHECK PASSED: accuracy {history[-1]['accuracy']:.3f} on {o.num_examples} fixed examples at step {reached_at}.")
        print("This shows the wiring can memorize; it is not evidence of generalization.")
    else:
        print(
            f"OVERFIT CHECK NOT PASSED within {max_steps} steps (best accuracy {outcome['best_accuracy']:.3f}). "
            "Investigate wiring, labels, gradients, optimizer, preprocessing before scaling up."
        )
    print(f"Saved {out}/overfit_result.json and overfit_curve.png")
    return 0 if outcome["success"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("env", "shapes", "step", "overfit"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=name in ("step", "overfit"), default=None)
        p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
        p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
        if name == "overfit":
            p.add_argument("--max-steps", type=int, default=None, help="override overfit.max_steps")
    args = parser.parse_args()
    return {"env": command_env, "shapes": command_shapes, "step": command_step, "overfit": command_overfit}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
