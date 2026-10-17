"""Length-generalization bonus, single-GPU CUDA entry point (e.g. one H100).

    python train_bonus_cloud.py --config configs/bonus_h100.yaml --set logging.run_name=bonus-h100-sinusoidal-s0
    python train_bonus_cloud.py --config configs/bonus_h100.yaml --set train.amp_dtype=bf16 --set logging.run_name=bonus-h100-bf16-s0
    python train_bonus_cloud.py --resume runs/<run>/checkpoints/last.pt

Same model, data, and loop as train_bonus_local.py (bonus_engine.run_bonus_training). There is no
DDP, custom attention kernels, or torch.compile.
"""

from __future__ import annotations

import argparse
import sys

import torch

from bonus_engine import run_bonus_training
from engine import config_for_run, resolve_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(f"CUDA is not available (torch {torch.__version__}, built with CUDA {torch.version.cuda}). Check nvidia-smi and the torch wheel.")
        return 1
    cfg, checkpoint = config_for_run(args.config, args.overrides, args.resume)
    if checkpoint is not None and checkpoint.get("task") != "bonus_decoder":
        parser.error("--resume expects a bonus checkpoint")
    if cfg.train.device not in ("cuda", "auto"):
        parser.error(f"train.device={cfg.train.device!r}; train_bonus_cloud.py runs on CUDA only")
    device = resolve_device("cuda", allowed=("cuda",))
    bf16 = torch.cuda.is_bf16_supported()
    print(f"GPU: {torch.cuda.get_device_name(device)} | CUDA {torch.version.cuda} | bf16 supported: {bf16}")
    if cfg.train.amp_dtype == "bf16" and not bf16:
        parser.error("train.amp_dtype=bf16 requested but this GPU does not support bfloat16")
    run_bonus_training(cfg, device, resume_checkpoint=checkpoint, resume_path=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
