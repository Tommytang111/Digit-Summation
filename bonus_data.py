"""Variable-length data for the length-generalization bonus (reuses dataset.py's machinery).

Each manifest is an ordinary dataset.SequenceManifest of ONE length L, produced by
dataset.generate_manifest with the same steps as the main task:
  1. digit ~ Uniform{0..9};
  2. image with probability data.image_probability, chosen independently per position;
  3. image operands drawn from that split's own MNIST pool;
  4. targets computed from text integers + MNIST labels.

    train_L01 .. train_L10           num_train_sequences / (#training lengths) rows each (uniform lengths)
    val_L01 .. val_L20               num_eval_per_length rows each; val pool images
    test_L01 .. test_L20             num_eval_per_length rows each; test pool images (official MNIST test)
    {val,test}_all_image_L01..L20    num_all_image_per_length rows each (every operand an image)

Cross-split duplicates: rows containing an image can never repeat across splits, because the
pools are disjoint and checked by hash. All-text rows depend only on digits, and at short lengths
they necessarily repeat (length 1 has only 10 possible text inputs). So bonus manifests are not
resampled to avoid them. The audit reports their overlap, and evaluation also scores "unseen
inputs only".

Batches are right-padded with an explicit Boolean padding_mask (dataset.collate_batch), so a
genuine zero operand (token 1) is never confused with padding (token 0 / mask True).
bonus_collate adds the teacher-forcing answer tensors.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from augmentations import derive_seed
from dataset import SequenceManifest, SourcePools, build_source_pools, collate_batch, generate_manifest
from tokenizer import OUT_PAD_ID, answer_ids, answer_text, teacher_forcing_pair

KINDS = {"natural": 0, "all_image": 1}
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}


def manifest_name(split: str, length: int, kind: str = "natural") -> str:
    """("val", 7) -> "val_L07"; ("test", 15, "all_image") -> "test_all_image_L15"."""
    return f"{split}{'' if kind == 'natural' else '_' + kind}_L{length:02d}"


def train_lengths(cfg) -> list[int]:
    return list(range(cfg.bonus.min_train_length, cfg.bonus.max_train_length + 1))


def eval_lengths(cfg) -> list[int]:
    return list(range(1, cfg.bonus.max_eval_length + 1))


def manifest_plan(cfg) -> list[dict]:
    """Generation specs in a fixed order (test, then val, then train), each with a recorded seed."""
    d, b = cfg.data, cfg.bonus
    base_seeds = {"train": d.train_seed, "val": d.val_seed, "test": d.test_seed}
    lengths = train_lengths(cfg)
    per_length, remainder = divmod(b.num_train_sequences, len(lengths))
    plan = []
    for split in ("test", "val"):
        for kind, count, probability in (("natural", b.num_eval_per_length, d.image_probability), ("all_image", b.num_all_image_per_length, 1.0)):
            for length in eval_lengths(cfg):
                seed = derive_seed(base_seeds[split], 7, KINDS[kind], length)
                plan.append({"name": manifest_name(split, length, kind), "pool_split": split, "count": count, "length": length, "p": probability, "seed": seed})
    for index, length in enumerate(lengths):
        count = per_length + (1 if index < remainder else 0)  # totals exactly num_train_sequences
        seed = derive_seed(base_seeds["train"], 7, KINDS["natural"], length)
        plan.append({"name": manifest_name("train", length), "pool_split": "train", "count": count, "length": length, "p": d.image_probability, "seed": seed})
    return plan


def build_bonus_data(cfg, arrays) -> tuple[SourcePools, dict[str, SequenceManifest]]:
    """Source pools (same seeds/fractions as the main task) + every bonus manifest, in memory."""
    d = cfg.data
    fractions = {"train": d.train_source_fraction, "val": d.val_source_fraction, "test": d.test_source_fraction}
    pools = build_source_pools(arrays, d.val_source_size, fractions, d.split_seed)
    manifests = {}
    for spec in manifest_plan(cfg):
        manifest = generate_manifest(
            name=spec["name"],
            pool_split=spec["pool_split"],
            pools=pools,
            arrays=arrays,
            num_sequences=spec["count"],
            sequence_length=spec["length"],
            image_probability=spec["p"],
            seed=spec["seed"],
            exclude_canonical=None,  # see module docstring: all-text overlap is reported, not resampled
        )
        manifests[spec["name"]] = manifest
    return pools, manifests


class MultiLengthDataset(Dataset):
    """Concatenates several DigitSumDatasets (one per length) behind one flat row index.

    Keys: an int global row, or (global_row, epoch) from engine.EpochShuffleSampler. The pair is
    forwarded as (local_row, epoch), so each sub-dataset reseeds augmentation per (epoch, row)
    from its own augment seed. Items gain "index" (global row) and "length".
    """

    def __init__(self, datasets: list):
        self.datasets = datasets
        self.offsets = np.cumsum([0] + [len(d) for d in datasets])  # [n_datasets + 1]
        self.row_lengths = np.concatenate([np.full(len(d), d.manifest.sequence_length) for d in datasets])

    @property
    def transform(self):
        """Shared training transform; augmentations.seed_worker reseeds it per worker."""
        return self.datasets[0].transform

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def locate(self, global_row: int) -> tuple[int, int]:
        """Global row -> (dataset index, local row)."""
        which = int(np.searchsorted(self.offsets, global_row, side="right") - 1)
        return which, int(global_row - self.offsets[which])

    def __getitem__(self, key) -> dict:
        global_row, epoch = (int(key[0]), int(key[1])) if isinstance(key, tuple) else (int(key), None)
        which, local_row = self.locate(global_row)
        item = self.datasets[which][(local_row, epoch) if epoch is not None else local_row]
        item["index"] = global_row
        item["length"] = int(self.row_lengths[global_row])
        return item


def bonus_collate(items: list[dict]) -> dict[str, torch.Tensor]:
    """dataset.collate_batch (right-padded operands + padding_mask) + teacher-forcing answer tensors.

    For a target sum 43: answer [BOS, "4", "3", EOS] -> decoder_input [BOS, "4", "3"],
    decoder_target ["4", "3", EOS]. Both are right-padded with OUT_PAD_ID to T = longest
    answer + 1 in the batch.

        decoder_input        int64 [B, T]
        decoder_target       int64 [B, T]   OUT_PAD at padding (ignored by the loss)
        decoder_padding_mask bool  [B, T]   True at padded decoder positions
        lengths              int64 [B]      number of real operands per row
    """
    batch = collate_batch(items)
    pairs = [teacher_forcing_pair(answer_ids(answer_text(int(item["target"])))) for item in items]
    width = max(len(decoder_input) for decoder_input, _ in pairs)
    decoder_input = torch.full((len(items), width), OUT_PAD_ID, dtype=torch.int64)
    decoder_target = torch.full((len(items), width), OUT_PAD_ID, dtype=torch.int64)
    decoder_padding_mask = torch.ones(len(items), width, dtype=torch.bool)
    for row, (inputs, targets) in enumerate(pairs):
        decoder_input[row, : len(inputs)] = torch.tensor(inputs)
        decoder_target[row, : len(targets)] = torch.tensor(targets)
        decoder_padding_mask[row, : len(inputs)] = False
    batch.update(
        decoder_input=decoder_input,
        decoder_target=decoder_target,
        decoder_padding_mask=decoder_padding_mask,
        lengths=(~batch["padding_mask"]).sum(dim=1),
    )
    return batch


def constructed_manifest(name: str, pool_split: str, digit_rows: list[list[int]], image_rows: list[list[bool]], arrays, pools, seed: int) -> SequenceManifest:
    """Hand-built diagnostic rows of ONE length (build one manifest per length), e.g. [3,3,3].

    digit_rows [N][L] are the operand digits and image_rows [N][L] mark image operands. Image
    operands use seeded random pool images with the right label, and targets come from the
    labels, exactly as in generate_manifest.
    """
    from dataset import SPLIT_TO_PARTITION, compute_targets

    rng = np.random.default_rng(seed)
    labels = arrays.labels[SPLIT_TO_PARTITION[pool_split]]
    pool = pools.ids[pool_split]
    length = len(digit_rows[0])
    if any(len(r) != length for r in digit_rows):
        raise ValueError("constructed_manifest expects rows of one length; build one manifest per length")
    digits = np.array(digit_rows, dtype=np.uint8)
    is_image = np.array(image_rows, dtype=bool)
    source = np.full(digits.shape, -1, dtype=np.int32)
    for digit in range(10):
        slots = is_image & (digits == digit)
        if slots.any():
            candidates = pool[labels[pool] == digit]
            source[slots] = candidates[rng.integers(0, len(candidates), int(slots.sum()))]
    targets = compute_targets(digits, is_image, source, labels).astype(np.int16)
    return SequenceManifest(name, pool_split, digits, is_image, source, targets, {"constructed": True, "seed": seed})
