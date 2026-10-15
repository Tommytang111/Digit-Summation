"""Local training entry point: Apple MPS (float32), or CPU for debugging.

    python train_local.py --config configs/debug.yaml                 # short smoke run
    python train_local.py --config configs/local.yaml                 # full local experiment
    python train_local.py --config configs/local.yaml --set augment.mode=none --set logging.run_name=local-noaug
    python train_local.py --resume runs/<run>/checkpoints/last.pt     # continue a run
    python train_local.py --config configs/debug.yaml --device cpu    # force CPU

All training logic lives in engine.py; this file only picks the device.
"""

from __future__ import annotations

import argparse
import sys

from engine import config_for_run, resolve_device, run_training


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", default=None, help="path to a last.pt/best.pt checkpoint")
    parser.add_argument("--device", default=None, choices=["auto", "mps", "cpu"], help="overrides train.device")
    args = parser.parse_args()

    cfg, checkpoint = config_for_run(args.config, args.overrides, args.resume)
    requested = args.device or cfg.train.device
    if requested == "cuda":
        parser.error("train.device=cuda: use train_cloud.py for CUDA runs")
    if cfg.train.amp_dtype != "none" or cfg.train.allow_tf32:
        parser.error("Local runs are float32 only (train.amp_dtype=none, train.allow_tf32=false)")
    device = resolve_device(requested, allowed=("mps", "cpu"))
    run_training(cfg, device, resume_checkpoint=checkpoint, resume_path=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
