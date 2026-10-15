"""MNIST source pools, compact sequence manifests, the PyTorch Dataset, and the collate function.

Pipeline (each stage only consumes the previous stage's output):

    official MNIST partitions (train 60k, test 10k; raw uint8 pixels + labels)
      -> source pools: disjoint image-ID sets for train / val / test       (build_source_pools)
      -> manifests: per sequence, digits + modality choice + source IDs    (generate_manifest)
      -> DigitSumDataset: reads a manifest row, fetches + transforms pixels on access
      -> collate_batch: stacks items into the model's tensor contract

A manifest never stores image tensors, only integer source IDs, so 20k sequences cost ~0.5 MB.

Why more sequences != more handwriting: every image operand is drawn from a fixed pool of MNIST
images. Generating 20k or 200k sequences re-combines the same ~50k training images (each image is
reused ~2x or ~20x); it adds new *combinations* and new sums, not new independent handwriting.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tokenizer import PAD_ID, encode_operand

PARTITIONS = ("train", "test")  # official MNIST partitions
SPLITS = ("train", "val", "test")  # our source pools
SPLIT_TO_PARTITION = {"train": "train", "val": "train", "test": "test"}
IMAGE_SHAPE = (28, 28)
NUM_DIGITS = 10
MODEL_INPUT_KEYS = ("images", "token_ids", "is_image", "padding_mask")  # everything else is metadata


# ----------------------------------------------------------------------------- raw MNIST


@dataclass
class MnistArrays:
    """Raw MNIST pixels and labels per official partition.

    images[partition]: uint8 [N, 28, 28], labels[partition]: int64 [N].
    Labels are used only to build pools/targets and for analysis, never as model inputs.
    """

    images: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]
    _hashes: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def hashes(self, partition: str) -> np.ndarray:
        """SHA-256 hex digest of each image's 784 raw bytes (cached)."""
        if partition not in self._hashes:
            self._hashes[partition] = hash_images(self.images[partition])
        return self._hashes[partition]


def load_mnist_arrays(root: str | Path, download: bool = False) -> MnistArrays:
    """Load both official partitions via torchvision.datasets.MNIST as raw uint8 arrays."""
    from torchvision.datasets import MNIST

    images, labels = {}, {}
    for partition in PARTITIONS:
        try:
            mnist = MNIST(str(root), train=(partition == "train"), download=download)
        except RuntimeError as error:
            raise RuntimeError(
                f"MNIST not found under {root!r}. Run `python prepare_data.py --config <cfg>` first "
                "(it downloads with torchvision)."
            ) from error
        images[partition] = mnist.data.numpy().astype(np.uint8, copy=False)  # [N, 28, 28]
        labels[partition] = mnist.targets.numpy().astype(np.int64)  # [N]
    return MnistArrays(images=images, labels=labels)


def hash_images(images: np.ndarray) -> np.ndarray:
    """Exact-content hashes: identical pixels <=> identical hash (near-duplicates are NOT caught)."""
    contiguous = np.ascontiguousarray(images, dtype=np.uint8)
    return np.array([hashlib.sha256(image.tobytes()).hexdigest() for image in contiguous])


# ----------------------------------------------------------------------------- source pools


@dataclass
class SourcePools:
    """Disjoint image-ID pools. ids[split] indexes into partition SPLIT_TO_PARTITION[split]."""

    ids: dict[str, np.ndarray]  # split -> sorted int64 indices
    report: dict  # duplicate resolution + count changes

    def pool_id(self) -> str:
        """Content identifier of the pools (stored in manifests and checkpoints)."""
        digest = hashlib.sha256()
        for split in SPLITS:
            digest.update(split.encode())
            digest.update(np.ascontiguousarray(self.ids[split], dtype=np.int64).tobytes())
        return digest.hexdigest()[:16]


def _largest_remainder(counts: np.ndarray, total: int) -> np.ndarray:
    """Allocate `total` items across classes proportionally to `counts`, summing exactly."""
    ideal = counts * (total / counts.sum())
    allocation = np.floor(ideal).astype(np.int64)
    leftover = total - allocation.sum()
    order = np.argsort(-(ideal - allocation), kind="stable")  # largest fractional parts first
    allocation[order[:leftover]] += 1
    return allocation


def stratified_split(ids: np.ndarray, labels: np.ndarray, holdout_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split `ids` into (remaining, holdout) with per-digit proportions preserved exactly."""
    rng = np.random.default_rng([seed, 0])
    per_digit = [ids[labels[ids] == digit] for digit in range(NUM_DIGITS)]
    holdout_counts = _largest_remainder(np.array([len(group) for group in per_digit]), holdout_size)
    remaining, holdout = [], []
    for group, count in zip(per_digit, holdout_counts):
        shuffled = rng.permutation(group)
        holdout.append(shuffled[:count])
        remaining.append(shuffled[count:])
    return np.sort(np.concatenate(remaining)), np.sort(np.concatenate(holdout))


def stratified_fraction(ids: np.ndarray, labels: np.ndarray, fraction: float, seed: int, stream: int) -> np.ndarray:
    """Keep round(fraction * n_d) images of each digit d (at least 1).

    Uses a fixed per-digit permutation and takes a prefix, so smaller fractions are nested
    subsets of larger ones (useful for data-efficiency experiments).
    """
    if fraction >= 1.0:
        return np.sort(ids)
    rng = np.random.default_rng([seed, 1, stream])
    kept = []
    for digit in range(NUM_DIGITS):
        group = ids[labels[ids] == digit]
        count = max(1, int(np.floor(fraction * len(group) + 0.5)))
        kept.append(rng.permutation(group)[:count])
    return np.sort(np.concatenate(kept))


def build_source_pools(
    arrays: MnistArrays,
    val_source_size: int,
    fractions: dict[str, float],
    seed: int,
) -> SourcePools:
    """Resolve exact duplicates, then build disjoint, stratified train/val/test image pools.

    Duplicate policy (exact SHA-256 matches, resolved before any sequence is generated):
      1. A development (official-train) image whose pixels also occur in the official test
         partition is removed from development; the official test pool is preserved.
      2. Within development, only the lowest index of each identical group is kept; a group
         with conflicting labels is removed entirely.
      3. Duplicates inside the official test partition are reported but kept.
    """
    dev_labels, test_labels = arrays.labels["train"], arrays.labels["test"]
    dev_hashes, test_hashes = arrays.hashes("train"), arrays.hashes("test")

    test_hash_set = set(test_hashes.tolist())
    duplicates_test = np.array([value in test_hash_set for value in dev_hashes.tolist()], dtype=bool)

    groups: dict[str, list[int]] = {}
    for index in np.flatnonzero(~duplicates_test):
        groups.setdefault(dev_hashes[index], []).append(int(index))
    dev_keep, internal_groups, internal_removed, conflict_groups, conflict_removed = [], 0, 0, 0, 0
    for members in groups.values():
        if len(members) == 1:
            dev_keep.append(members[0])
            continue
        internal_groups += 1
        if len(set(dev_labels[members].tolist())) > 1:
            conflict_groups += 1
            conflict_removed += len(members)
        else:
            dev_keep.append(min(members))
            internal_removed += len(members) - 1
    dev_ids = np.sort(np.array(dev_keep, dtype=np.int64))

    _, test_counts = np.unique(test_hashes, return_counts=True)
    if len(dev_ids) <= val_source_size:
        raise ValueError(f"val_source_size={val_source_size} leaves no training images ({len(dev_ids)} total)")

    train_full, val_full = stratified_split(dev_ids, dev_labels, val_source_size, seed)
    test_full = np.arange(len(test_labels), dtype=np.int64)
    ids = {
        "train": stratified_fraction(train_full, dev_labels, fractions["train"], seed, stream=0),
        "val": stratified_fraction(val_full, dev_labels, fractions["val"], seed, stream=1),
        "test": stratified_fraction(test_full, test_labels, fractions["test"], seed, stream=2),
    }

    def per_digit(split: str) -> list[int]:
        labels = arrays.labels[SPLIT_TO_PARTITION[split]]
        return np.bincount(labels[ids[split]], minlength=NUM_DIGITS).tolist()

    report = {
        "official_counts": {"train": int(len(dev_labels)), "test": int(len(test_labels))},
        "dev_images_duplicating_test_removed": int(duplicates_test.sum()),
        "dev_internal_duplicate_groups": internal_groups,
        "dev_internal_duplicates_removed": internal_removed,
        "dev_conflicting_label_groups": conflict_groups,
        "dev_conflicting_label_images_removed": conflict_removed,
        "test_internal_duplicate_groups_kept": int((test_counts > 1).sum()),
        "test_internal_duplicate_extra_copies_kept": int((test_counts - 1).sum()),
        "dev_pool_after_dedup": int(len(dev_ids)),
        "proposed_train_val": [50_000, 10_000],
        "full_train_pool": int(len(train_full)),
        "full_val_pool": int(len(val_full)),
        "train_shortfall_vs_50000": int(50_000 - len(train_full)),
        "fractions": dict(fractions),
        "final_counts": {split: int(len(ids[split])) for split in SPLITS},
        "final_per_digit": {split: per_digit(split) for split in SPLITS},
    }
    return SourcePools(ids=ids, report=report)


def save_source_pools(pools: SourcePools, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "source_pools.npz", **{f"{split}_ids": pools.ids[split] for split in SPLITS})
    with open(directory / "source_pools.json", "w") as handle:
        json.dump({"pool_id": pools.pool_id(), "report": pools.report}, handle, indent=2)


def load_source_pools(directory: Path) -> SourcePools:
    with np.load(directory / "source_pools.npz") as data:
        ids = {split: data[f"{split}_ids"].astype(np.int64) for split in SPLITS}
    with open(directory / "source_pools.json") as handle:
        report = json.load(handle)["report"]
    return SourcePools(ids=ids, report=report)


# ----------------------------------------------------------------------------- manifests


@dataclass
class SequenceManifest:
    """Compact, reproducible description of arithmetic sequences (no pixels).

    digits:       uint8 [N, L]  sampled digit value per operand (label metadata)
    is_image:     bool  [N, L]  True = operand shown as an MNIST image, False = as text
    source_index: int32 [N, L]  MNIST index in partition SPLIT_TO_PARTITION[pool_split], -1 for text
    targets:      int16 [N]     sum computed from text integers + MNIST labels
    """

    name: str
    pool_split: str  # which source pool images come from: "train" | "val" | "test"
    digits: np.ndarray
    is_image: np.ndarray
    source_index: np.ndarray
    targets: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def partition(self) -> str:
        return SPLIT_TO_PARTITION[self.pool_split]

    @property
    def num_sequences(self) -> int:
        return int(self.digits.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.digits.shape[1])

    def manifest_id(self) -> str:
        """Content hash: identical manifests <=> identical ID (recorded in checkpoints)."""
        digest = hashlib.sha256(self.pool_split.encode())
        for array, dtype in ((self.digits, np.uint8), (self.is_image, np.bool_), (self.source_index, np.int32)):
            digest.update(np.ascontiguousarray(array, dtype=dtype).tobytes())
        return digest.hexdigest()[:16]

    def _derived(self, name: str, digits, is_image, source_index, targets) -> "SequenceManifest":
        return SequenceManifest(name, self.pool_split, digits, is_image, source_index, targets, {"derived_from": self.name})

    def subset(self, rows: np.ndarray, name: str | None = None) -> "SequenceManifest":
        rows = np.asarray(rows)
        return self._derived(
            name or f"{self.name}_subset", self.digits[rows], self.is_image[rows], self.source_index[rows], self.targets[rows]
        )

    def with_all_text(self) -> "SequenceManifest":
        """ORACLE substitution: every image operand replaced by its true digit as text.

        Uses ground-truth labels, so it is a diagnostic only. It also changes the modality
        distribution (every row becomes all-text, which is rare in training).
        """
        return self._derived(
            f"{self.name}_oracle_text",
            self.digits.copy(),
            np.zeros_like(self.is_image),
            np.full_like(self.source_index, -1),
            self.targets.copy(),
        )

    def permuted(self, seed: int) -> "SequenceManifest":
        """Same rows with operand positions shuffled independently per row (seeded)."""
        rng = np.random.default_rng(seed)
        order = np.argsort(rng.random(self.digits.shape), axis=1)  # one random permutation per row
        take = lambda array: np.take_along_axis(array, order, axis=1)  # noqa: E731
        return self._derived(
            f"{self.name}_permuted", take(self.digits), take(self.is_image), take(self.source_index), self.targets.copy()
        )


def compute_targets(digits: np.ndarray, is_image: np.ndarray, source_index: np.ndarray, partition_labels: np.ndarray) -> np.ndarray:
    """Sum = text operands' integer values + image operands' MNIST labels.

    Also asserts that every image's MNIST label equals the digit that was sampled for its slot.
    """
    values = digits.astype(np.int64).copy()
    image_labels = partition_labels[source_index[is_image]]
    if not np.array_equal(image_labels, values[is_image]):
        raise AssertionError("An image operand's MNIST label disagrees with its sampled digit")
    values[is_image] = image_labels
    return values.sum(axis=1)


def input_keys(manifest: SequenceManifest, partition_hashes: np.ndarray) -> tuple[list[str], list[str]]:
    """Per row: (exact key, permutation-canonical key) describing the *complete model input*.

    Text operand -> "t<digit>"; image operand -> "i<sha256 of its pixels>".
    The canonical key sorts the operands, so it also matches reordered rows. Because this model
    has no positional embeddings it is permutation-invariant: a reordered row is effectively the
    same input.
    """
    exact, canonical = [], []
    for digits_row, image_row, source_row in zip(manifest.digits, manifest.is_image, manifest.source_index):
        tokens = [
            f"i{partition_hashes[source]}" if is_image else f"t{digit}"
            for digit, is_image, source in zip(digits_row.tolist(), image_row.tolist(), source_row.tolist())
        ]
        exact.append("|".join(tokens))
        canonical.append("|".join(sorted(tokens)))
    return exact, canonical


def _sample_rows(rng, count, length, image_probability, pool_by_digit):
    digits = rng.integers(0, NUM_DIGITS, size=(count, length)).astype(np.uint8)  # step 1: uniform digits
    is_image = rng.random((count, length)) < image_probability  # step 2: independent modality
    source_index = np.full((count, length), -1, dtype=np.int32)
    for digit in range(NUM_DIGITS):  # step 3: image of that digit from this split's own pool
        slots = is_image & (digits == digit)
        if slots.any():
            pool = pool_by_digit[digit]
            source_index[slots] = pool[rng.integers(0, len(pool), size=int(slots.sum()))]
    return digits, is_image, source_index


def generate_manifest(
    name: str,
    pool_split: str,
    pools: SourcePools,
    arrays: MnistArrays,
    num_sequences: int,
    sequence_length: int,
    image_probability: float,
    seed: int,
    exclude_canonical: set[str] | None = None,
    max_rounds: int = 1_000,
) -> SequenceManifest:
    """Generate sequences, resampling rows whose canonical input appears in an earlier split.

    Steps per operand: digit ~ Uniform{0..9}; image with prob `image_probability`; if image, a
    uniformly chosen pool image with that label; target = text integers + MNIST labels.
    """
    partition = SPLIT_TO_PARTITION[pool_split]
    labels = arrays.labels[partition]
    hashes = arrays.hashes(partition)
    pool_ids = pools.ids[pool_split]
    pool_by_digit = [pool_ids[labels[pool_ids] == digit] for digit in range(NUM_DIGITS)]
    if image_probability > 0 and any(len(pool) == 0 for pool in pool_by_digit):
        raise ValueError(f"Pool {pool_split!r} lacks some digit; increase its source fraction")

    rng = np.random.default_rng(seed)
    digits, is_image, source_index = _sample_rows(rng, num_sequences, sequence_length, image_probability, pool_by_digit)
    exclude = exclude_canonical or set()
    resampled = 0
    for _ in range(max_rounds):
        probe = SequenceManifest(name, pool_split, digits, is_image, source_index, np.zeros(len(digits), np.int16))
        _, canonical = input_keys(probe, hashes)
        clashes = np.array([key in exclude for key in canonical], dtype=bool)
        if not clashes.any():
            break
        resampled += int(clashes.sum())
        new_digits, new_is_image, new_source = _sample_rows(
            rng, int(clashes.sum()), sequence_length, image_probability, pool_by_digit
        )
        digits[clashes], is_image[clashes], source_index[clashes] = new_digits, new_is_image, new_source
    else:
        raise RuntimeError(f"{name}: could not avoid cross-split duplicates in {max_rounds} rounds")

    targets = compute_targets(digits, is_image, source_index, labels).astype(np.int16)
    metadata = {
        "seed": int(seed),
        "pool_split": pool_split,
        "image_probability": float(image_probability),
        "num_sequences": int(num_sequences),
        "sequence_length": int(sequence_length),
        "rows_resampled_to_avoid_cross_split_duplicates": resampled,
    }
    return SequenceManifest(name, pool_split, digits, is_image, source_index, targets, metadata)


def save_manifests(manifests: dict[str, SequenceManifest], directory: Path, extra: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    index = {"extra": extra, "manifests": {}}
    for name, manifest in manifests.items():
        np.savez(
            directory / f"{name}.npz",
            digits=manifest.digits,
            is_image=manifest.is_image,
            source_index=manifest.source_index,
            targets=manifest.targets,
        )
        index["manifests"][name] = {**manifest.metadata, "manifest_id": manifest.manifest_id(), "file": f"{name}.npz"}
    with open(directory / "manifests.json", "w") as handle:
        json.dump(index, handle, indent=2)


def load_manifests(directory: Path) -> dict[str, SequenceManifest]:
    index_path = directory / "manifests.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No manifests in {directory}. Run `python prepare_data.py --config <cfg>` first.")
    with open(index_path) as handle:
        index = json.load(handle)
    manifests = {}
    for name, meta in index["manifests"].items():
        with np.load(directory / meta["file"]) as data:
            manifest = SequenceManifest(
                name=name,
                pool_split=meta["pool_split"],
                digits=data["digits"],
                is_image=data["is_image"],
                source_index=data["source_index"],
                targets=data["targets"],
                metadata=meta,
            )
        if manifest.manifest_id() != meta["manifest_id"]:
            raise ValueError(f"Manifest {name} content does not match its recorded ID; files were modified")
        manifests[name] = manifest
    return manifests


# ----------------------------------------------------------------------------- dataset + collate


class DigitSumDataset(Dataset):
    """Turns manifest rows into model inputs on access; pixels are fetched and transformed lazily.

    __getitem__ returns (shapes for one sequence of length L):
        images:    float32 [L, 1, 28, 28]  transformed pixels; text slots hold a 0.0 placeholder
        token_ids: int64   [L]             digit token IDs; image slots hold PAD_ID
        is_image:  bool    [L]
        target:    int                     the sum (training label)
        digits:    int64   [L]             metadata for analysis only (never fed to the model)
        index:     int                     row index in the manifest
    """

    def __init__(self, manifest: SequenceManifest, partition_images: np.ndarray, transform):
        if partition_images.dtype != np.uint8 or partition_images.shape[1:] != IMAGE_SHAPE:
            raise ValueError("partition_images must be uint8 [N, 28, 28] raw MNIST pixels")
        self.manifest = manifest
        self.partition_images = partition_images
        self.transform = transform  # uint8 [28, 28] -> float32 [28, 28]; owns its RNG

    def __len__(self) -> int:
        return self.manifest.num_sequences

    def __getitem__(self, index: int) -> dict:
        digits_row = self.manifest.digits[index]
        is_image_row = self.manifest.is_image[index]
        source_row = self.manifest.source_index[index]
        length = len(digits_row)

        images = torch.zeros(length, 1, *IMAGE_SHAPE, dtype=torch.float32)
        token_ids = torch.full((length,), PAD_ID, dtype=torch.int64)
        for position in range(length):
            if is_image_row[position]:
                raw_pixels = self.partition_images[source_row[position]]  # uint8 [28, 28]
                images[position, 0] = torch.from_numpy(self.transform(raw_pixels))
            else:
                token_ids[position] = encode_operand(int(digits_row[position]))  # 7 -> "7" -> 8
        return {
            "images": images,
            "token_ids": token_ids,
            "is_image": torch.from_numpy(is_image_row.astype(np.bool_)),
            "target": int(self.manifest.targets[index]),
            "digits": torch.from_numpy(digits_row.astype(np.int64)),
            "index": int(index),
        }


def collate_batch(items: list[dict]) -> dict[str, torch.Tensor]:
    """Stack variable-length items into the batch tensor contract (L = longest item).

    images:       float32 [B, L, 1, 28, 28]
    token_ids:    int64   [B, L]   PAD_ID at image slots *and* at padding
    is_image:     bool    [B, L]   False at padding
    padding_mask: bool    [B, L]   True only at real padding (built from lengths, never token IDs)
    targets:      int64   [B]
    digits:       int64   [B, L]   metadata; -1 at padding
    indices:      int64   [B]      metadata
    """
    batch_size = len(items)
    max_length = max(len(item["token_ids"]) for item in items)
    images = torch.zeros(batch_size, max_length, 1, *IMAGE_SHAPE, dtype=torch.float32)
    token_ids = torch.full((batch_size, max_length), PAD_ID, dtype=torch.int64)
    is_image = torch.zeros(batch_size, max_length, dtype=torch.bool)
    padding_mask = torch.ones(batch_size, max_length, dtype=torch.bool)
    digits = torch.full((batch_size, max_length), -1, dtype=torch.int64)
    for row, item in enumerate(items):
        length = len(item["token_ids"])
        images[row, :length] = item["images"]
        token_ids[row, :length] = item["token_ids"]
        is_image[row, :length] = item["is_image"]
        padding_mask[row, :length] = False
        digits[row, :length] = item["digits"]
    return {
        "images": images,
        "token_ids": token_ids,
        "is_image": is_image,
        "padding_mask": padding_mask,
        "targets": torch.tensor([item["target"] for item in items], dtype=torch.int64),
        "digits": digits,
        "indices": torch.tensor([item["index"] for item in items], dtype=torch.int64),
    }


# ----------------------------------------------------------------------------- convenience


@dataclass
class PreparedData:
    arrays: MnistArrays
    pools: SourcePools
    manifests: dict[str, SequenceManifest]

    def data_ids(self) -> dict:
        """Identifiers stored in checkpoints so evaluation can verify it uses the same data."""
        return {
            "source_pool_id": self.pools.pool_id(),
            "manifest_ids": {name: manifest.manifest_id() for name, manifest in self.manifests.items()},
        }

    def dataset(self, name_or_manifest, transform) -> DigitSumDataset:
        manifest = self.manifests[name_or_manifest] if isinstance(name_or_manifest, str) else name_or_manifest
        return DigitSumDataset(manifest, self.arrays.images[manifest.partition], transform)


def load_prepared_data(data_cfg) -> PreparedData:
    """Load MNIST arrays + frozen pools + manifests written by prepare_data.py."""
    directory = Path(data_cfg.manifest_dir)
    arrays = load_mnist_arrays(data_cfg.root, download=False)
    pools = load_source_pools(directory)
    manifests = load_manifests(directory)
    return PreparedData(arrays=arrays, pools=pools, manifests=manifests)
