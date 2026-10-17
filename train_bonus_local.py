"""Length-generalization bonus, local entry point: Apple MPS (float32), or CPU for debugging.

    python train_bonus_local.py --config configs/bonus_debug.yaml                     # smoke run
    python train_bonus_local.py --config configs/bonus_local.yaml --set logging.run_name=bonus-sinusoidal-s0
    python train_bonus_local.py --config configs/bonus_local.yaml --set bonus.architecture=items_only_no_positions --set logging.run_name=bonus-nopos-s0
    python train_bonus_local.py --resume runs/<run>/checkpoints/last.pt

All training logic lives in bonus_engine.py (shared with train_bonus_cloud.py); this file only
picks the device.
"""

from __future__ import annotations

import argparse
import sys

from bonus_engine import run_bonus_training
from engine import config_for_run, resolve_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", default=None, help="path to a bonus last.pt/best.pt checkpoint")
    parser.add_argument("--device", default=None, choices=["auto", "mps", "cpu"], help="overrides train.device")
    args = parser.parse_args()

    cfg, checkpoint = config_for_run(args.config, args.overrides, args.resume)
    if checkpoint is not None and checkpoint.get("task") != "bonus_decoder":
        parser.error("--resume expects a bonus checkpoint; use train_local.py for main-task checkpoints")
    requested = args.device or cfg.train.device
    if requested == "cuda":
        parser.error("train.device=cuda: use train_bonus_cloud.py for CUDA runs")
    if cfg.train.amp_dtype != "none" or cfg.train.allow_tf32:
        parser.error("Local runs are float32 only (train.amp_dtype=none, train.allow_tf32=false)")
    run_bonus_training(cfg, resolve_device(requested, allowed=("mps", "cpu")), resume_checkpoint=checkpoint, resume_path=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
