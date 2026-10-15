"""Single-GPU CUDA entry point (e.g. one H100). Same model, data, and training loop as local runs.

    python train_cloud.py --config configs/h100.yaml
    python train_cloud.py --config configs/h100.yaml --set train.amp_dtype=bf16 --set logging.run_name=h100-bf16
    python train_cloud.py --resume runs/<run>/checkpoints/last.pt

No DDP, FlashAttention, or torch.compile. All training logic lives in engine.py.
"""

from __future__ import annotations

import argparse
import sys

import torch

from engine import config_for_run, resolve_device, run_training


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. Check `nvidia-smi` and that torch was installed with a CUDA wheel:")
        print(f"  torch {torch.__version__}, built with CUDA {torch.version.cuda}")
        return 1
    cfg, checkpoint = config_for_run(args.config, args.overrides, args.resume)
    if cfg.train.device not in ("cuda", "auto"):
        parser.error(f"train.device={cfg.train.device!r}; train_cloud.py runs on CUDA only")
    device = resolve_device("cuda", allowed=("cuda",))
    name = torch.cuda.get_device_name(device)
    bf16 = torch.cuda.is_bf16_supported()
    print(f"GPU: {name} | capability {torch.cuda.get_device_capability(device)} | CUDA {torch.version.cuda} | bf16 supported: {bf16}")
    if cfg.train.amp_dtype == "bf16" and not bf16:
        parser.error("train.amp_dtype=bf16 requested but this GPU does not support bfloat16")
    print(
        f"Precision: {'bf16 autocast' if cfg.train.amp_dtype == 'bf16' else 'float32'}"
        f"{' + TF32 matmuls' if cfg.train.allow_tf32 else ''} | batch {cfg.train.batch_size} x accum {cfg.train.grad_accum_steps} "
        f"| workers {cfg.train.num_workers}"
    )
    run_training(cfg, device, resume_checkpoint=checkpoint, resume_path=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
