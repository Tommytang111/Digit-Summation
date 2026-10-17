# Digit Summation: MNIST images + text digits → the sum, as text

Each example is a sequence of **ten operands**. Each position is independently either a
**text integer 0–9** or an **MNIST image** of a digit. A small model built from scratch (shared CNN +
token embeddings + 2-layer Transformer encoder) classifies the sum into **91 classes (0–90)** and
returns the predicted class as a string, e.g. `"47"`.

> **Status (2026-10-16).** The main-task pipeline is implemented and verified: unit tests, data
> inspection, leakage audit, shape and gradient checks, and a bounded tiny-overfit check. One
> full training run on the 100,000-sequence data (`full-s0`, a single seed) reached **95.87%
> validation** and **95.50% test** exact-sum accuracy. That run used the *original* image encoder
> (GELU, average pooling, no BatchNorm). The defaults have since changed to ReLU + max pooling +
> BatchNorm, and no sum-model run with the new encoder exists yet. The section
> [What has actually been run](#what-has-actually-been-run) lists every check and result.
> **Length-generalization bonus:** implemented (autoregressive answer decoder, lengths 1–10 → 1–20)
> and verified with tests, structural checks, tiny-overfit checks, and smoke runs. **No full bonus
> training run exists yet**, so there are no bonus results. See
> [Length-generalization bonus](#length-generalization-bonus).

---

## Contents
1. [Quick start](#quick-start-staged-commands)
2. [Directory layout](#directory-layout)
3. [Setup and tested versions](#setup-and-tested-versions)
4. [Pipeline walkthrough](#pipeline-walkthrough)
   - [Data: source pools, manifests, leakage audit](#1-data-source-pools-and-manifests)
   - [Tokenization and the tensor contract](#2-tokenization-and-the-tensor-contract)
   - [Model](#3-model)
   - [Augmentation](#4-augmentation-albumentations)
   - [Training: epochs vs optimizer steps, checkpoints](#5-training)
   - [Evaluation and diagnostics](#6-evaluation)
   - [Logging and Weights & Biases](#7-logging-and-weights--biases)
5. [Experiments to run next](#experiments-to-run-next)
6. [What has actually been run](#what-has-actually-been-run): checks, training runs, results
7. [Troubleshooting](#troubleshooting)
8. [Limitations](#limitations)
9. [Length-generalization bonus](#length-generalization-bonus): decoder, positions, diagnostics, commands, status
10. [References](#references)

---

## Quick start (staged commands)

Run everything from the repo root with the project venv (here `../.torch`). The `PY` variable is
shorthand; use `python` if the venv is activated.

```bash
cd /Users/tommy/Projects/Digit-Summation
PY=../.torch/bin/python

# 1. Install dependencies and inspect devices (torch is already in .torch; see Setup for other machines)
$PY -m pip install -r requirements.txt
$PY sanity_checks.py env

# 2. Download MNIST, build the tiny stratified debug subset, run the leakage audit
$PY prepare_data.py --config configs/debug.yaml
$PY prepare_data.py --config configs/debug.yaml --audit-only        # re-audit what is on disk

# 3. Inspect raw + augmented examples, tokenization, targets, distributions
$PY inspect_data.py --config configs/debug.yaml                     # -> runs/inspect-debug/
$PY inspect_data.py --config configs/local.yaml --rows 16           # full data -> runs/inspect-local/
#    options: --config (required), --set KEY=VALUE, --out DIR, --rows N (default 8; rows are
#    drawn with a fixed seed from the whole training manifest, so every run shows the same rows)

# 4. Test the dataset and collate function in isolation (synthetic data, no download)
$PY -m pytest tests/test_dataset.py tests/test_tokenizer.py
$PY -m pytest                                                       # whole suite (88 tests, ~40 s)
$PY -m pytest CNN-encoder-test                                      # the encoder benchmark's own 7 tests

# 5. Image encoder / token embedding / Transformer shape checks
$PY sanity_checks.py shapes

# 6. Real forward/backward optimizer updates on one debug batch
$PY sanity_checks.py step --config configs/debug.yaml

# 7. Overfit 64 fixed examples, aug + dropout off, <= 300 optimizer updates
$PY sanity_checks.py overfit --config configs/debug.yaml
$PY sanity_checks.py overfit --config configs/debug.yaml --set overfit.image_probability=1.0   # image path only

# 8. Short local training job (debug data, 60 updates) + validation evaluation
$PY train_local.py --config configs/debug.yaml --set logging.run_name=debug-short
$PY evaluate.py --checkpoint runs/debug-short/checkpoints/best.pt --split val

# 9. Full local experiment (explicit; up to 78,000 optimizer updates, early stopping may end it
#    sooner; full-s0 took ~51 min on the M4 Pro with the previous encoder). local.yaml logs to W&B
#    online: run `wandb login` first, or add --set logging.wandb_mode=offline
$PY prepare_data.py --config configs/local.yaml                     # already built; verifies frozen manifests
$PY train_local.py  --config configs/local.yaml --set logging.run_name=full-affine-s0

# 10. Evaluate the validation-selected checkpoint and generate deliverables
$PY evaluate.py --checkpoint runs/full-affine-s0/checkpoints/best.pt --split val
$PY evaluate.py --checkpoint runs/full-affine-s0/checkpoints/best.pt --split test   # final number, once

# 11. Same project/config/manifests on an H100 (see "Running on an H100")
rsync -av --exclude runs --exclude data/MNIST ./ user@h100-host:~/Digit-Summation/
#   on the H100 host:
#   python prepare_data.py --config configs/h100.yaml    # downloads MNIST, VERIFIES manifests are identical
#   python train_cloud.py  --config configs/h100.yaml --set logging.run_name=h100-affine-s0
```

---

## Directory layout

```
Digit-Summation/
├── config.py              typed dataclass config; YAML loading; unknown-key rejection; --set overrides
├── configs/
│   ├── debug.yaml         5% source subsets, 2000/500/500 sequences, <=60 updates, W&B disabled
│   ├── local.yaml         full data (data/manifests/full), 100000/20000/20000 sequences, MPS float32,
│   │                      batch 256, 200 epochs, warmup + cosine, early stopping, W&B online
│   └── h100.yaml          identical data + optimization to local.yaml, on CUDA with 8 workers + pinned memory
├── tokenizer.py           fixed vocabulary: PAD=0, "0".."9" -> 1..10; integer -> string -> token IDs
├── dataset.py             MNIST arrays, hashing, source pools, manifests, DigitSumDataset, collate_batch
├── augmentations.py       Albumentations pipelines (none / affine / affine_elastic), worker seeding
├── model.py               ImageEncoder (CNN) + DigitSumTransformer (embeddings, encoder, pooling, head)
├── engine.py              devices, seeding, sampler, train loop, evaluate, checkpoints, resume
├── experiment_logging.py  metrics.jsonl / config.json / summary.json + optional W&B
├── visualization.py       all matplotlib figures (galleries, curves, distributions, breakdowns)
├── prepare_data.py        download -> pools -> frozen manifests -> executable leakage audit
├── inspect_data.py        example grids, augmentation previews, normalization + elastic checks, stats
├── sanity_checks.py       env | shapes | step | overfit
├── train_local.py         thin MPS/CPU entry point
├── train_cloud.py         thin single-GPU CUDA entry point
├── evaluate.py            explicit checkpoint evaluation (val or test) + diagnostics + figures + W&B eval run
├── compare_runs.py        tabulates runs; verifies they share pools and eval manifests (main and bonus)
├── bonus_model.py         bonus: sinusoidal positions, SumAnswerModel (3 variants), greedy generation
├── bonus_data.py          bonus: per-length manifests, MultiLengthDataset, bonus_collate (answer tensors)
├── bonus_engine.py        bonus: teacher-forced training loop, free-generation scoring, checkpoints
├── prepare_bonus_data.py  bonus: frozen manifests (train 1-10, val/test 1-20) + leakage audit
├── inspect_bonus_data.py  bonus: answer tokens, batch contract, sum ranges by length
├── bonus_checks.py        bonus: shapes | step | overfit | structure
├── train_bonus_local.py   bonus: thin MPS/CPU entry point
├── train_bonus_cloud.py   bonus: thin single-GPU CUDA entry point
├── evaluate_bonus.py      bonus: explicit free-generation evaluation at every length 1-20
├── configs/bonus_{debug,local,h100}.yaml   bonus configurations (local and h100 mirrored)
├── tests/                 145 focused tests (88 main task + 57 bonus) on synthetic fixtures
├── CNN-encoder-test/      supervised digit-classifier reference for the image encoder: implied
│                          sum ceiling, encoder variant benchmark (own README and 7 tests)
├── requirements.txt       non-torch dependencies pinned to tested versions
├── pytest.ini, .gitignore, .env.example (placeholders only)
├── data/                  (gitignored) MNIST download + data/manifests/{debug,main,full,bonus,bonus-debug}/
├── runs/                  (gitignored) one directory per run / inspection / overfit check
└── wandb/                 (gitignored) logs written by the wandb CLI (e.g. `wandb sync`)
```

---

## Setup and tested versions

Tested on macOS 26.5, Apple M4 Pro (24 GB), Python 3.12.15:

| package | version | note |
|---|---|---|
| torch / torchvision | 2.14.1 / 0.29.1 | MPS built + available |
| albumentations | 2.0.8 | the requested package; **not** AlbumentationsX |
| albucore / opencv-python-headless | 0.0.24 / 5.0.0.93 | pulled in by albumentations |
| numpy / matplotlib / PyYAML | 2.5.4 / 3.11.2 / 6.0.3 | |
| wandb / pytest | 0.30.0 / 9.1.1 | |

**Installing PyTorch per platform.** PyTorch wheels are platform-specific, so torch is not in
`requirements.txt`:

- **macOS / Apple silicon:** `pip install torch==2.14.1 torchvision==0.29.1`. The standard
  wheels include MPS. There is no CUDA on Macs.
- **Linux + NVIDIA (H100):** `pip install torch==2.14.1 torchvision==0.29.1`. The default
  Linux wheel of 2.14.1 depends on CUDA 13.0 (`cuda-toolkit==13.0.3`) and needs a driver that
  supports CUDA 13. For an older driver, use the CUDA 12.6 build:
  `pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126`.
  `cu126`, `cu130`, and `cu132` wheels exist for this version, and all support Hopper (sm_90).
  Check the "CUDA Version" printed by `nvidia-smi` and pick a build that is ≤ it.

**About albumentations.** Version 2.0.8 is the newest release of the original MIT-licensed
`albumentations` package on PyPI. Upstream has archived it, and development continues as
AlbumentationsX (AGPL-3.0 or a commercial license). This project uses the requested original
package. Every API call used here (`Affine(fill=...)`, `ElasticTransform(noise_distribution=...)`,
`Compose(seed=...)`, `set_random_seed`) was checked against the installed 2.0.8 source.

---

## Pipeline walkthrough

### 1. Data: source pools and manifests

```
official MNIST train (60,000) ─┬─ exact-duplicate resolution ─┬─ train pool (50,000) ─ train manifest (100,000 seq)
                               │                              └─ val pool   (10,000) ─ val manifest   (20,000 seq)
official MNIST test (10,000) ──┴──────────────────────────────── test pool  (10,000) ─ test manifest  (20,000 seq)
```

Sequence counts are those of `configs/local.yaml` and `configs/h100.yaml`. Three manifest sets
exist on disk, each with its own leakage audit:

| directory | train / val / test sequences | source pools | used by |
|---|---|---|---|
| `data/manifests/debug` | 2,000 / 500 / 500 | 5% subsets (2,500 / 500 / 501 images) | `debug.yaml` |
| `data/manifests/main` | 20,000 / 2,000 / 5,000 | full | the earlier 20k runs (`local-affine-s0`, `local-20261015-*`); no current config |
| `data/manifests/full` | 100,000 / 20,000 / 20,000 | full | `local.yaml`, `h100.yaml`, `full-s0`, `CNN-encoder-test/` |

**Source pools come first, and arithmetic examples come only after the pools are isolated.**
`build_source_pools` hashes every raw image (SHA-256 of its 784 bytes) and resolves exact
duplicates before splitting:

1. A development (official-train) image with the same pixels as any official-test image is
   removed from development, so the official test pool is preserved intact.
2. Within development, only the lowest index of each identical group is kept. A group with
   conflicting labels is removed entirely.
3. Duplicates inside the official test set are reported and kept.

**Actual result on MNIST: there are no exact duplicates at all**, whether within train, within
test, or between them. The proposed **stratified 50,000 / 10,000** split therefore holds
exactly. Validation gets exactly 1/6 of each digit (per-class largest-remainder rounding), and the
test pool is all 10,000 official test images. The audit records these counts (and would record
any changes) in `data/manifests/*/source_pools.json`.

**Source fractions** (`data.train/val/test_source_fraction`) keep a stratified fraction of each
pool, at least one image per digit. Smaller fractions are **nested subsets** of larger ones,
because each takes a prefix of a fixed per-digit permutation. A 10% data-efficiency run therefore
sees a subset of the 50% run's images.

**Manifests** are compact and reproducible. They store integer IDs, never pixels:

| array | dtype, shape | meaning |
|---|---|---|
| `digits` | uint8 [N, 10] | sampled digit per operand (*label metadata, not a model input*) |
| `is_image` | bool [N, 10] | modality choice per operand |
| `source_index` | int32 [N, 10] | MNIST index for image operands, −1 for text |
| `targets` | int16 [N] | the sum |

Generation follows the assignment's order. For each operand:
1. Draw the digit uniformly from 0–9.
2. Independently choose image (p = 0.5) or text.
3. If image, draw a uniformly random image *of that digit* from **that split's own pool**.

The target is then computed from the text integers plus the **MNIST labels** looked up via
`source_index`, and the generator asserts that each image's label equals the sampled digit. The
full manifests (100k/20k/20k plus diagnostic sets) take 9.1 MB, and the dataset fetches and
transforms pixels lazily on access.

**Frozen evaluation data.** Every manifest has its own recorded seed (train 1001, val 2002,
test 3003, diagnostics derived from 4004) and a content ID. Re-running `prepare_data.py`
regenerates everything and checks that it is **identical** to what is on disk. If a config change
would alter the data, it refuses to overwrite unless you pass `--overwrite`. Checkpoints store the
pool and manifest IDs, and `evaluate.py` refuses to evaluate against different data. Manifests are
generated in the order test → val → train. A row whose input already exists in an earlier
manifest is *resampled*, which keeps test fixed and adjusts train instead.

**Why more sequences ≠ more handwriting.** Every image operand comes from a fixed pool. The
20,000-sequence `main` manifest has ~100k image slots but uses only **43,269 distinct training
images** (2.32 uses each). The other ~13.5% of the 50k pool is never used, which matches the
Poisson expectation e^−2. The 100,000-sequence `full` manifest has ~500k image slots and uses
**49,997 of the 50,000** images, about 10 times each. Going from 20k to 100k sequences therefore
added new *combinations and sums* and the last ~13% of the pool, but no other new handwriting.
Image diversity is controlled by the **source fraction**, not by the sequence count.

**Target distribution is not uniform.** A sum of 10 uniform digits has an exact,
bell-shaped distribution (computed by convolution in `inspect_data.py`). It peaks at **45 with
p = 0.0432**, while P(sum ∈ {0, 90}) = 2.0 × 10⁻¹⁰. The 100k training manifest has its majority
at 45 (4.36%), and **18 classes never occur in it at all** (0–9 and 83–90). For the 20k `main`
manifest these were 45 (4.62%) and 24 classes. `inspect_data.py --config configs/local.yaml`
writes `runs/inspect-local/distributions.png`, which also shows the image-count distribution
against Binomial(10, 0.5).

#### Leakage audit (`prepare_data.py`, executable, exit code 1 on failure)

Hard checks (**76/76 pass** for the debug, main, and full data):
- Source pools are disjoint as (partition, index) IDs.
- Every manifest's image operands come from its own split's pool, and text slots have no image source.
- No exact raw-pixel hash is shared between pools.
- **No exact complete-input duplicate** across any pair of manifests, including the diagnostic
  sets. The complete input is the ordered list of `text:digit` / `image:sha256` operands.
- **No permutation-equivalent input** across manifests (the same multiset of operands in another order).
- Targets equal text integers + MNIST labels, and every image label matches its sampled digit.
- Real dataset items carry **PAD tokens at every image slot**, so image labels never leak into inputs.

Reused digit values and reused target sums across splits are **legitimate** and are reported as
such, not flagged. **All-text rows** are the only rows that *can* collide across splits, since
they contain no images. The full manifests contain 85 (train), 14 (val), and 22 (test) all-text
rows. Generation resampled 5 train rows, 5 val rows, 2 test-diagnostic rows, and 34
val-diagnostic rows that would have matched an earlier manifest.

**What the audit cannot do:**
- **Near-duplicates** (a shifted, rescaled, or re-encoded copy of an image) differ in hash and are
  not detected.
- **Writers:** MNIST does not ship writer IDs, so the train/val split is **not writer-independent**,
  and this project makes no writer-independence claim. (MNIST's own documentation describes its
  official train and test writer sets as disjoint; we rely on that only implicitly, by keeping the
  official test partition as the test pool.)
- **Permutation equivalence** is blocked because *this* model has no positional embeddings and is
  exactly order-invariant, so a reordered row is effectively the same input. A future
  order-sensitive model would still treat reordered rows as near-duplicates of the task.

### 2. Tokenization and the tensor contract

Numeric operands go through an explicit string step (`tokenizer.py`):

```
integer 7  ──number_to_string──▶  "7"  ──tokenize──▶  [8]  ──nn.Embedding(11, 128, padding_idx=0)──▶  128-d vector
```

| token | `<pad>` | "0" | "1" | … | "9" |
|---|---|---|---|---|---|
| ID | 0 | 1 | 2 | … | 10 |

Token IDs are **categorical indices**, not arithmetic values: ID 8 is "the token whose string is
7", and its vector is learned. There is no scalar n/9 feature. **Digit 0 is token 1, not PAD.**

Batch contract (built by `collate_batch`, verified in `tests/test_dataset.py`):

| key | dtype, shape | notes |
|---|---|---|
| `images` | float32 [B, L, 1, 28, 28] | normalized pixels; text slots hold a finite 0.0 placeholder |
| `token_ids` | int64 [B, L] | PAD_ID at image slots **and** at padding |
| `is_image` | bool [B, L] | False at padding |
| `padding_mask` | bool [B, L] | True **only** for real padding, built from lengths |
| `targets` | int64 [B] | sums 0–90 |
| `digits`, `indices` | int64 | metadata for analysis; never passed to the model |

L = 10 for the main task, so `padding_mask` is all False there. Collate already pads variable
lengths, which the bonus needs. An image slot has `token_ids == PAD_ID` but is **not** padding,
which is why padding is never derived from token IDs. `model.forward(images, token_ids,
is_image, padding_mask)` accepts only these four inputs, so labels cannot be passed in by
accident.

### 3. Model

```
images [B,10,1,28,28] ─ select real image slots ─▶ [N_img,1,28,28]
   └─ ImageEncoder: conv3x3(1→32)·BatchNorm·ReLU·maxpool2 ─▶ [N_img,32,14,14]
                    conv3x3(32→64)·BatchNorm·ReLU·maxpool2 ─▶ [N_img,64,7,7] ─ flatten ─▶ [N_img,3136]
                    Linear ─▶ [N_img,128] ─ LayerNorm ─▶ scatter back into [B,10,128]
token_ids [B,10] ─ nn.Embedding(11,128,padding_idx=0) ─▶ [B,10,128]
torch.where(is_image, image_features, text_features) ─▶ [B,10,128]
  + modality embedding (0 = text, 1 = image)          ─▶ [B,10,128]
2 × TransformerEncoderLayer(d=128, heads=4, ff=512, dropout=0.1, GELU, pre-LN, batch_first)
  with src_key_padding_mask = padding_mask, no causal mask ─▶ [B,10,128]
final LayerNorm ─▶ masked sum over the 10 operands ─▶ [B,128] ─ Linear ─▶ logits [B,91]
answer text = str(argmax(logits))
```

Parameters: **830,907** (image encoder 420,704; encoder layers 396,544; head 11,739; token
embedding 1,408; modality embedding 256; final norm 256). The original GELU encoder without
BatchNorm had 830,811. `python sanity_checks.py shapes`
prints every shape transition and confirms that the step-by-step result equals `forward`.

Design choices, and why:
- **Attention heads.** d_model = 128 is split into 4 heads of 32 dims. Each head computes its own
  attention pattern over the 10 operands, and the outputs are concatenated back to 128. The
  **feedforward** block is a per-operand MLP 128 → 512 → 128 (GELU) after attention.
- **No causal mask.** The encoder is bidirectional, so every operand attends to every real
  operand. `tests/test_model.py` checks that changing the *last* operand changes position 0's
  output. Padding is excluded as attention keys (`src_key_padding_mask`) and from pooling
  (`masked_fill`).
- **No positional embeddings.** Addition is order-invariant and each operand is one digit, so
  positions carry no information. Without them the whole model is permutation-invariant. On
  `full-s0`, the test-time permutation check measured prediction agreement 1.0000 on both val
  and test, with max logit differences ≤ 9.9 × 10⁻⁵ (float rounding).
- **Independently initialized layers.** `nn.TransformerEncoder(layer, num_layers)` builds its
  stack with `copy.deepcopy` (verified in the installed torch source, `_get_clones`), so all its
  layers start with identical weights. This model builds a `ModuleList` of separately constructed
  layers instead, which a test checks.
- **Masked sum pooling, then a linear head.** The pooled vector is Σᵢ h(operandᵢ). If the network
  learned a per-operand "value" direction v, a linear head can decode the integer, because
  −(v − k)² = 2kv − k² − v², and the v² term is the same for every class k. This shows that a
  linear head *can* represent the decoding. It is **not** hard-coded arithmetic: the encoder and
  head must learn such features from sum supervision alone.
- **Zero-initialized head.** Summing 10 LayerNorm outputs gives a large pooled vector. With the
  default random head, the measured initial loss was **9.31**, against ln 91 = **4.51** for
  uniform guessing, and the first gradient norm was ~100. Zero-initializing the head starts
  training exactly at 4.5109. Side effect: on the very first update only the head receives
  gradient, because dL/dpooled = Wᵀ(p − y) = 0. `sanity_checks.py step` reports both updates.
- **LayerNorm at the end of the CNN** keeps image features on the same scale as the N(0, 1)
  token embeddings. Both measured std ≈ 1.37 per slot at init, including the modality embedding.
- **Modality embedding (optional, on by default).** A learned vector per modality tags which
  encoder produced each slot, so attention can treat CNN features and token embeddings
  differently. Disable it with `--set model.use_modality_embedding=false` to ablate.
- **CNN encoder: ReLU + max pooling + BatchNorm (the defaults).**
  - Set by `model.cnn_activation` (`relu`/`gelu`), `model.cnn_pooling` (`max`/`avg`), and
    `model.cnn_batch_norm`. BatchNorm follows each conv, and the conv bias is dropped because
    BatchNorm's shift replaces it.
  - **Why these defaults:** chosen from a 3-seed standalone digit benchmark in
    `CNN-encoder-test/`. Validation digit accuracy was 0.9924 ± 0.0002 for this encoder vs
    0.9915 ± 0.0005 for the original GELU + avg-pool encoder without BatchNorm. The implied
    validation sum ceiling was 0.9629 vs 0.9588.
  - **Not yet measured:** a sum-model run with this encoder.
  - **BatchNorm and placeholders:** the CNN only sees gathered real image slots, so text
    placeholders never enter its batch statistics (tested). In eval mode it uses running
    averages, so evaluation stays permutation-invariant.
  - **Older checkpoints:** a checkpoint from before these options rebuilds as what it was.
    `engine.load_checkpoint` fills in GELU without BatchNorm, and max pooling if pooling is
    unrecorded. Your earlier runs (e.g. `full-s0`: GELU / avg / no BN) still evaluate correctly.
- **Gather/scatter for images.** The CNN runs only on real image slots, never on placeholders,
  and is skipped entirely for all-text batches. Tests check that gradients reach the CNN only when
  images are present and reach digit-token embeddings only when text is present.
- Loss is `F.cross_entropy` applied directly to the logits.

### 4. Augmentation (Albumentations)

Each image goes through: raw uint8 [28, 28] → (training only) geometric ops on uint8, **fill = 0
(black, the MNIST background), before normalization** → `Normalize(mean = 0.1307, std = 0.3081)`
**exactly once, last** → float32 [28, 28].

| mode | operations (training only) |
|---|---|
| `none` | Normalize only (validation and test always use this) |
| `affine` (default) | `Affine`: rotate ±10°, translate ±2 px (integer), per-axis scale 0.9–1.1, per-axis shear ±5°, bilinear interpolation, constant black border, p = 1 |
| `affine_elastic` | affine + `ElasticTransform(alpha=12, sigma=4, noise_distribution="uniform")`, p = 0.5 |

No flips, 90° rotations, or crops. These would change digit identity (6 ↔ 9) or cut strokes. The
settings are *starting values to validate*, previewed for all ten digits in
`runs/inspect-*/augmentation_preview.png`.

Verified (in `inspect_data.py` and `tests/test_augmentations.py`):
- Grayscale arrays stay [28, 28], and the output is float32.
- Normalization runs exactly once: a black pixel maps to −0.42421 and a white one to 2.82149. On
  the 50k training pool, the normalized mean/std is 0.0001 / 1.0003.
- The fill is black: a fully white image gains pixels at exactly the normalized-black value.
- Evaluation preprocessing is deterministic.

**Elastic parameters were measured, not copied.** In albumentations 2.0.8, with
`noise_distribution="uniform"`, the displacement field is α × GaussianBlur(uniform(−1, 1), σ)
pixels. `measure_elastic_displacement` measures it by warping a coordinate image:

| α (σ = 4) | RMS displacement / axis | 95th pct. magnitude | max | pixels sampled outside image |
|---|---|---|---|---|
| 4 | 0.17 px | 0.42 px | 0.84 px | 0% |
| 8 | 0.34 px | 0.83 px | 1.69 px | 0% |
| **12 (default)** | **0.51 px** | **1.25 px** | 2.53 px | 0% |
| 20 | 0.85 px | 2.08 px | 4.22 px | 0% |
| 34 | 1.44 px | 3.51 px | 7.18 px | 0.19% |

α = 12 was chosen as "mild" (≈ half a pixel RMS). Note that the library's *default*
`noise_distribution="gaussian"` divides the noise by its maximum, so the same α would produce a
different strength under that setting.

**Sources.**
- Affine distortions of digits (translation, scaling, squeezing, shearing) come from the
  classic MNIST work of LeCun, Bottou, Bengio & Haffner (1998).
- Elastic distortions were introduced later by Simard, Steinkraus & Platt (2003). Their MNIST
  setting is commonly quoted as σ = 4, α = 34. I could not machine-extract that value from the PDF
  in this session, so check §2 of the paper. This project **does not reproduce** their setup: the
  library and parameter conventions differ, and no comparison was run.

**Seeding: augmentation is a pure function of (seed, epoch, row).**
- Albumentations' Compose owns its RNG, so `np.random.seed` does **not** affect it.
- The training sampler yields `(row, epoch)` pairs. Before transforming a row's images,
  `DigitSumDataset` reseeds the transform with `derive_seed(augment_seed, epoch, row)`.
- **Consequence:** `num_workers`, `persistent_workers`, batch size, and resume do not change which
  augmentation a row gets. Runs that differ only in loader settings see **identical training
  data**, which makes loader speed comparisons controlled.
- This is not the classic "reseed per sample" bug, which reseeds with *one constant* and gives
  every image the same transform. Here every (epoch, row) has its own seed.
- **Verified:** a test checks that 0 workers, 2 workers, and 2 persistent workers produce
  bit-identical batches over two epochs. Two 40-update debug trainings, with 0 and with 2
  persistent workers, logged identical training losses at every step.
- **Cost:** reseeding adds ~28 µs per sequence (`affine`, ~11%) or ~20 µs (`affine_elastic`,
  ~5%), measured on the full data.
- **Shared generator.** albumentations' own `Compose.set_random_seed(s)` gives *every child* an
  identically seeded generator, so `Affine` and `ElasticTransform` would replay the same random
  numbers. Under per-row reseeding that would become a systematic correlation. `seed_compose`
  instead gives Compose and all children one shared generator pair, and a test checks this.
- `seed_worker` still reseeds each worker's global RNGs and transform. That matters only for
  dataset accesses that carry no epoch, because without it every worker replays the same stream
  (measured under albumentations 2.0.8).
- **Limit:** GPU float arithmetic is not bit-reproducible. Two runs with an *identical* config and
  0 workers ended 40 updates with weights differing by up to 3.5e-7, the same size as a 0- vs
  2-worker pair (2.5e-7). Validation loss can differ in the last float32 digit (2.4e-7).

### 5. Training

**Epochs vs optimizer steps.**
- A *micro-batch* is one forward/backward pass on `batch_size` sequences.
- An *optimizer step* is one `optimizer.step()`, covering `grad_accum_steps` micro-batches.
  `global_step` counts **only** optimizer steps and is the x-axis of every plot and log.
- An *epoch* is one pass over the fixed training manifest. With `drop_last` it has
  ⌊100000 / 256⌋ = 390 micro-batches under `local.yaml`; an incomplete accumulation group at
  epoch end is discarded. 200 epochs × 390 = 78,000 planned optimizer steps.
- `train.max_steps` caps optimizer steps, `train.epochs` caps passes, and whichever comes first
  ends the run.

Settings:
- **Optimizer:** AdamW, lr 3e-4. Weight decay 0.01 applies only to ≥ 2-D Linear/Conv/attention
  weights, not to biases, norms, or embeddings.
- **Schedule:** `local.yaml` and `h100.yaml` use `warmup_cosine`: linear warmup over 500 steps,
  then cosine decay to 0 at the last planned step. The code default (and `debug.yaml`) is a
  constant learning rate.
- **Gradients:** clipped to norm 1.0.
- **Shuffling:** a per-epoch permutation seeded by (seed, epoch).
- **Validation and selection:** validation runs every `eval_every_steps` and at the end.
  **`best.pt` is selected by validation accuracy** (ties broken by validation loss), and
  `last.pt` is saved at every evaluation. The test split is never touched during training.
- **Early stopping (optional), on validation accuracy:**
  - `train.early_stopping_patience: N` ends training after N validations in a row without a
    new highest validation accuracy. A gain of `early_stopping_min_delta` or less (in accuracy
    units, e.g. 0.001 = 0.1 points) doesn't count. `null` (the default) turns it off;
    `local.yaml` and `h100.yaml` set 15.
  - Patience is counted in validations, so 15 × `eval_every_steps: 1000` = 15,000 updates.
  - The counter is saved in checkpoints. A resumed run continues the count, and a run that
    already stopped stays stopped unless you raise the patience. `summary.json` records
    `stopped_early` and `planned_optimizer_steps`.
  - Accuracy is used rather than loss because, on `full-s0`, validation loss reached its minimum
    (0.618) at step 29k while accuracy kept improving (0.929 at 29k, 0.945 at 47k, best 0.959
    at 69k). A loss-based patience of 10 would have stopped at 39k. Replayed over all 78 of
    `full-s0`'s validations, the accuracy rule with patience 15 never triggers: the longest
    stretch without a new best was 9 validations. (`full-s0` itself predates early stopping and
    ran all 200 epochs.)
  - With `warmup_cosine`, an early stop also skips the low-learning-rate end of the schedule.

**Checkpoints** (`runs/<run>/checkpoints/{best,last}.pt`) contain:
- model, optimizer, and scheduler state; the scaler slot is `None`, since float32 and bf16 need no
  GradScaler
- the full config and the vocabulary
- source-pool and manifest IDs
- epoch, global step, and the best metric
- Python/NumPy/torch/MPS/CUDA RNG states
- library versions

They load with `torch.load(weights_only=True)`, the safe unpickler, and always onto the CPU first,
so a checkpoint saved on MPS loads on CUDA or CPU. This is tested.

**Resume:** `train_local.py --resume runs/<run>/checkpoints/last.pt [--set train.max_steps=N]`.
- **Restored:** data order, per-row augmentation, model, optimizer, scheduler, and RNG states.
- **Mid-epoch:** the run skips the batches already consumed. The remaining rows get exactly the
  augmentations an uninterrupted run would have used, up to GPU float nondeterminism.
- Local metrics continue in the same run directory. A W&B run on resume starts a new W&B run.

**Devices.**
- *Local (`train_local.py`):** MPS float32, or CPU (`--device cpu`). No CUDA calls, and pinned
  memory is honored only on CUDA. `PYTORCH_ENABLE_MPS_FALLBACK` is never set; if you set it, a
  loud warning is printed, because it hides missing MPS kernels by running them on the CPU. An
  unsupported op raises an explicit error that suggests `--device cpu`.
  `sanity_checks.py env` probes every op of the model (forward, backward, AdamW) on CPU and MPS;
  both pass.
- *Cloud (`train_cloud.py`):* single CUDA GPU, the same `engine.run_training`. Optional
  `train.amp_dtype=bf16` is enabled only after `torch.cuda.is_bf16_supported()`; TF32 is opt-in.
  No DDP, FlashAttention, or `torch.compile`.
- **Throughput** is timed after `torch.mps.synchronize()` / `torch.cuda.synchronize()`, measured
  end to end including data loading, and validation time is excluded. Measured on the M4 Pro:
  a median of 6,772 seq/s for `full-s0` (batch 256, 4 persistent workers, previous encoder).
- **Data-loader workers.**
  - `train.num_workers` sets how many processes build batches in parallel. With
    `train.persistent_workers: true`, they stay alive across epochs and validation passes, so
    they don't restart and re-import every time.
  - Neither setting changes the data (see Seeding).
  - `local.yaml` uses 4 persistent workers; `debug.yaml` uses 0, which is easiest to debug.
  - The validation loader shares these settings.

### 6. Evaluation

`evaluate.py --checkpoint ... --split {val,test}` (test only when you ask for it; optional
`--device`, `--out`, `--wandb-mode`) writes to `runs/<run>/eval_<split>_<checkpoint>/`:
`metrics.json`, `predictions.csv` (answers as text), `metrics_summary.png` (accuracy, macro F1,
and MAE next to the majority baseline, one panel per metric), `examples.png`, `failures.png`,
`breakdown.png`, and `stress.png`. It refuses to run if the pools or manifests on disk differ
from the checkpoint's. It also logs the evaluation to W&B (see
[Logging and Weights & Biases](#7-logging-and-weights--biases)). The report keeps several
populations **separate**:

| section | what it is |
|---|---|
| **main** | the naturally sampled split: exact-sum accuracy, macro F1, MAE, loss, n; accuracy by number of image operands (0–10) and by sum range (0–9 … 80–90), each with n |
| **majority baseline** | always answer the most frequent training sum (45 for the full data: **4.33%** test accuracy); its accuracy, macro F1, and MAE are reported |
| text-only set | 1,000 all-text sequences (no handwriting needed) |
| all-image set | 1,000 all-image sequences from the same split's pool |
| permutation test | operands shuffled per row; agreement with the original predictions + max logit difference |
| **ORACLE substitution** | every image replaced by its *true* digit token. This uses ground-truth labels, so it is a diagnostic upper reference, **not** a result. It also changes the modality mix: every row becomes all-text, which is rare (0.1%) in training. |
| sums unseen in training | count and accuracy of eval rows whose sum never occurs in the training manifest (full data: 1 val row, 0 test rows) |
| **stress cases** | constructed all-0 / all-9 rows (text, image, mixed). Sums 0 and 90 essentially never occur in training (p = 1e-10), so failures here are expected and reported apart from natural results |

**Macro F1** treats each sum class equally, unlike accuracy:
- For each class c, F1_c = 2·TP_c / (#predicted c + #true c). Equivalently, it is the harmonic
  mean of precision and recall for "the sum is c".
- Macro F1 is the plain mean over the classes that appear in the targets **or** the predictions,
  which is sklearn's `average="macro"` convention (a test checks the match).
  - A class that is only predicted, or never predicted, scores 0.
  - Classes absent from both are skipped.
- **How to read it next to accuracy:**
  - Accuracy is dominated by the common sums near 45. Macro F1 gives a rare sum like 12 (a
    handful of validation rows) the same weight as 45, so it is sensitive to the tails and
    noisier.
  - Classes that never occur in training (0–9 and 83–90 for the full data) can only score 0 when
    they appear in an evaluation set.
  - Always guessing 45 scores almost nothing: one class with a small F1, averaged with zeros.
- Checkpoint selection still uses validation **accuracy**; macro F1 is reported alongside it.
- Diagnostic sets (text-only, all-image, permutation, oracle) report macro F1 too, each over its
  own population.

Galleries are **reproducibly sampled** (fixed seeds). `examples.png` shows a random draw that
includes successes and failures alike, and `failures.png` shows a random draw from the errors
only. Neither is cherry-picked.

### 7. Logging and Weights & Biases

Every run always writes local, machine-readable records, so no network or credentials are needed:

- `runs/<run>/config.json` holds the resolved config, seeds, device, parameter counts,
  source/sequence counts, and data IDs. A resumed session writes `config_resumed.json` instead.
- `runs/<run>/metrics.jsonl` has one JSON object per event, each with `optimizer_step`,
  `wall_time_s`, and a fractional `epoch`.
  - Training: `train/loss`, `train/accuracy`, `train/macro_f1`, `train/lr`,
    `train/samples_per_sec`, `train/steps_per_sec`, `train/grad_norm_preclip`. These are averaged
    over the sequences of each logging interval. `train/macro_f1` uses per-class counts
    accumulated on the device over that interval.
  - Validation: `val/loss`, `val/accuracy`, `val/macro_f1`, `val/mae`, each over the full
    validation set, plus `val/best_accuracy` and `val/validations_without_improvement` (the
    early-stopping counter).
  - `summary.json` adds `best_val_macro_f1`, the macro F1 of the accuracy-selected checkpoint.
- `runs/<run>/loss_curves.png` plots training and validation loss, accuracy, and macro F1 **against
  optimizer steps**. Training points are interval means in train mode (dropout and augmentation
  on); validation points are full eval-mode passes.
- `train_examples.png` (what the model sees) and `val_predictions_best.png`.
- `summary.json`.

W&B modes (`logging.wandb_mode`):
- `disabled`: wandb is never imported. This is the code default and `debug.yaml`'s setting.
- `offline`: writes to `runs/<run>/wandb/`; upload later with `wandb sync`.
- `online`: streams to wandb.ai. **`local.yaml` and `h100.yaml` use this**, with project
  `digit-summation` and entity `neoscith-solo`, so they need `wandb login` (or `WANDB_API_KEY`)
  and network access.

```bash
wandb login                                   # once; or export WANDB_API_KEY in your shell (never commit it)
$PY train_local.py --config configs/local.yaml --set logging.run_name=full-affine-s0          # online
$PY train_local.py --config configs/local.yaml --set logging.wandb_entity=<your-entity> \
    --set logging.run_name=full-affine-s0                                                     # your own entity
$PY train_local.py --config configs/local.yaml --set logging.wandb_mode=offline \
    --set logging.run_name=full-affine-s0                                                     # no network
wandb sync runs/<run>/wandb/offline-run-*     # upload an offline run later
```

The x-axis is defined explicitly: `wandb.define_metric("*", step_metric="optimizer_step")`.
Images are logged once at start (training examples) and once at the end (predictions and
curves). There is no per-step image or gradient logging. Checkpoint artifact upload is opt-in
(`--set logging.upload_checkpoints=true`). The project never reads, prints, or logs the API key,
and logged configs pass through a key-name filter.

**Evaluations in W&B.** Each `evaluate.py` call is logged as its own W&B run:
- **Name and group:** the run is named `<run>-eval_<split>_<checkpoint>` (e.g.
  `full-s0-eval_test_best`), has job type `eval`, and sits in the training run's W&B group.
  Training runs are grouped under their run name too; `full-s0` itself predates grouping.
- **Metrics:** every number in `metrics.json` goes to the run's summary under `<split>/...`:
  `test/accuracy`, `test/macro_f1`, `test/mae`, `test/majority_baseline/accuracy`,
  `test/by_image_count/5/accuracy`, `test/by_sum_range/40-49/accuracy`,
  `test/diagnostics/oracle_substitution/accuracy`, `test/stress_cases/all_9_image/accuracy`, and
  so on (126 values for `full-s0`'s test evaluation). They show as columns in the W&B runs table.
- **Figures:** the five evaluation figures are logged as images.
- **Mode:** it follows the checkpoint's `logging.wandb_mode`, so `full-s0`-style runs log online and
  debug runs don't log at all. Override it with `--wandb-mode {disabled,offline,online}`; an
  offline evaluation is written to `runs/<run>/eval_<split>_<checkpoint>/wandb/`.

---

## Experiments to run next

All of these are commands, **not results**. Each comparison uses the same frozen manifests
(`data/manifests/full`). `compare_runs.py` verifies that before printing a table.

```bash
# Augmentation: controlled comparison (identical pools + val/test manifests). Use >= 3 seeds;
# single-seed differences can be noise. Do not assume augmentation helps. Run names must be new:
# train_local.py refuses an existing runs/<name> (full-affine-s0 is also quick-start step 9).
for seed in 0 1 2; do
  for mode in none affine affine_elastic; do
    $PY train_local.py --config configs/local.yaml --set augment.mode=$mode --set train.seed=$seed \
        --set logging.run_name=full-$mode-s$seed
  done
done
$PY compare_runs.py runs/full-*-s0 runs/full-*-s1 runs/full-*-s2

# Data efficiency: fewer independent handwriting samples (nested subsets). This needs its own
# manifest_dir because it changes the pools.
$PY prepare_data.py --config configs/local.yaml --set data.train_source_fraction=0.1 --set data.manifest_dir=data/manifests/full-src10
$PY train_local.py  --config configs/local.yaml --set data.train_source_fraction=0.1 --set data.manifest_dir=data/manifests/full-src10 --set logging.run_name=full-src10

# Ablations
$PY train_local.py --config configs/local.yaml --set model.use_modality_embedding=false --set logging.run_name=full-nomod

# Loader speed: only num_workers / persistent_workers change, so the training data is identical.
# A capped max_steps keeps each run short. Don't run other GPU jobs at the same time.
for w in 0 2 4 6 8; do
  p=$([ $w -gt 0 ] && echo true || echo false)
  $PY train_local.py --config configs/local.yaml --set train.max_steps=2000 \
      --set train.num_workers=$w --set train.persistent_workers=$p \
      --set logging.wandb_mode=disabled --set logging.run_name=speed-w$w
done
$PY compare_runs.py runs/speed-w*    # seq/s + train time per run; lists every differing setting
```

`compare_runs.py` prints:
- **seq/s**: the median end-to-end training throughput. Validation passes and the first slow
  interval don't skew it.
- **train time**: total wall time, including worker startup and validation passes.
- **Config settings that differ**, which should be only `train.num_workers` and
  `train.persistent_workers`.

To isolate the cost of starting workers, also run `w=4` with `persistent_workers=false`.

**Running on an H100.** Copy the project *with its frozen manifests* (`data/manifests/`) and
install torch for CUDA (see Setup). Then run `prepare_data.py --config configs/h100.yaml`: it
downloads MNIST and **verifies** the manifests regenerate identically (it already prints
"verified" on the Mac). Then run `train_cloud.py --config configs/h100.yaml`.

`h100.yaml` and `local.yaml` are independent files, and nothing checks that they stay in sync. For
an MPS-vs-H100 comparison to be controlled, give both the same data, model, seed, batch size, lr,
epochs and precision. Worker count and pinned memory can differ. Any other difference (batch
size, bf16, TF32, epochs) changes the experiment, so name such runs accordingly
(e.g. `h100-bf16-s0`). `compare_runs.py` still checks that compared runs used the same data.
Nothing in this repo provisions cloud resources.

---

## What has actually been run

All of the following ran on the M4 Pro (MPS) between 2026-10-15 and 2026-10-16.

### Checks

| check | outcome |
|---|---|
| `sanity_checks.py env` | all versions found. Op probe (forward + backward + AdamW on mixed / all-text / all-image / padded batches) **OK on CPU and MPS**. `PYTORCH_ENABLE_MPS_FALLBACK` not set |
| `pytest` | **88 passed** in ~40 s (re-run 2026-10-16; the DataLoader-worker test starts real worker processes). `pytest CNN-encoder-test`: 7 passed. Coverage: tokenizer, config, pools/dedup/fractions, manifests/targets, cross-split resampling, dataset/collate contract, audit (passes on clean data, catches injected leaks), augmentation (normalize once, black fill, per-(epoch, row) seeding, shared child generator, identical batches for 0/2/2-persistent workers), model (shapes, masks/padding, all-image/all-text, gradient branches, permutation invariance, bidirectionality, independent layers, CPU vs MPS), engine (macro F1 vs sklearn + hand cases + device-side counts on CPU/MPS, checkpoint round trip, MPS load, RNG, sampler, weight-decay groups, accumulation step counting, resume, early stopping on val accuracy incl. resume, tolerant checkpoint-state loading), evaluation reporting (metric flattening for W&B, no W&B when disabled, summary plot incl. old metrics without macro F1) |
| `prepare_data.py` (debug, main, full) | **audit 76/76 passed** for all three. No exact duplicates in MNIST; pools are 50,000 / 10,000 / 10,000 (debug: 2,500 / 500 / 501). A re-run verified identical data (full: re-verified 2026-10-16), and a changed config was refused |
| `inspect_data.py` (debug, main, full) | figures written. Normalization, pixel ranges, tokenization trace, and distributions as reported above. The full-data numbers come from a 2026-10-16 run with `configs/local.yaml`; the `runs/inspect-local/` on this machine is older and shows the 20k `main` data until step 3 is re-run |
| `sanity_checks.py shapes` | every boundary as in the diagram. Step-by-step == forward. Layers independent |
| `sanity_checks.py step` | initial loss 4.5109 = ln 91. Update 1: head-only gradient (expected). Update 2: finite, non-zero gradient in every component. PAD row gradient 0. Batch loss 4.51 → 3.95 after two updates (a wiring check only) |
| `sanity_checks.py overfit` (64 fixed mixed examples, aug off, dropout 0, lr 1e-3, cap 300) | **passed: 100% at update 30** (fixed-set loss 0.056) |
| same, all-image fixed set (`image_probability=1.0`) | **passed: 100% at update 50**, so the image path can memorize too |
| same, at the training LR 3e-4 | **passed: 100% at update 40** |
| overfit again with the current ReLU / max / BatchNorm encoder (lr 1e-3) | mixed set **100% at update 40** (fixed-set loss 0.013); all-image set **100% at update 70** |
| `train_local.py` debug, 60 updates | pipeline OK (checkpoints, curves, galleries, summary). It found and fixed one bug: `torch.__version__` in checkpoints broke `weights_only` loading. Two identical runs logged identical losses. Throughput ≈ 1,800 seq/s end to end |
| `evaluate.py --split val` on that debug checkpoint | all outputs written. Permutation agreement 1.0000, max logit difference 1.4e-6 |
| resume from `last.pt` mid-epoch | continued at update 60 and stopped at the epoch cap (62) |
| `num_workers` 0 vs 2 (persistent), 40 debug updates each | identical training losses at every logged step. Final weights differ by ≤ 2.5e-7, the same size as two runs of an identical 0-worker config (≤ 3.5e-7), i.e. GPU float nondeterminism, not data. The same checkpoint evaluated with 0 vs 2 workers gives bit-identical inputs, logits, and loss |
| W&B **offline** debug run (20 updates) | offline run written, with the `wandb sync` command printed. No key fields in the local records |
| W&B **online** | used by `full-s0` and the three `local-20261015-*` runs |
| W&B evaluation runs (2026-10-16) | `evaluate.py --wandb-mode offline` on a debug checkpoint: the offline run holds all 153 metrics, matching `metrics.json`, plus the five figures. `full-s0`'s existing test evaluation was then logged online from its saved `metrics.json` (no re-evaluation) as `full-s0-eval_test_best`: 126 metrics + 5 figures |
| `CNN-encoder-test/digit_cnn.py`, 3 seeds × 2 encoders | supervised digit accuracy and implied sum ceilings; the ReLU / max / BatchNorm variant was adopted as the default. See [CNN-encoder-test/README.md](CNN-encoder-test/README.md) |

**The tiny-overfit successes show only that the wiring can memorize.** The debug run's numbers
(validation accuracy 6.6% after 60 updates, vs. the 5.0% majority baseline on that split) are a
pipeline smoke test and **say nothing about convergence**.

### Training runs

All are seed 0, `affine` augmentation, lr 3e-4, on MPS, and use the *previous* image encoder
(GELU, no BatchNorm). They differ in several settings at once, so they are not a controlled
comparison.

| run | data | batch / epochs / schedule | pooling | updates | best val acc (step) | training time |
|---|---|---|---|---|---|---|
| `local-affine-s0` | main (20k) | 64 / 40 / constant | max | 12,480 | 0.757 (12,480) | 365 s |
| `local-20261015-134910` | main (20k) | 64 / 100 / constant | max | 31,200 | 0.874 (29,250) | 948 s |
| `local-20261015-143936` | main (20k) | 128 / 100 / warmup_cosine | avg | 15,600 | 0.870 (15,000) | 347 s |
| `local-20261015-151200` | main (20k) | 128 / 200 / warmup_cosine | avg | 31,200 | 0.930 (31,000) | 686 s |
| **`full-s0`** | full (100k) | 256 / 200 / warmup_cosine | avg | 78,000 | **0.9587 (69,000)** | 3,069 s |

Training time includes validation passes. `full-s0` ran at a median 6,772 seq/s with 4
persistent workers.

### Results: `full-s0` (best checkpoint, step 69,000)

| | val | test |
|---|---|---|
| **exact-sum accuracy** (n = 20,000) | **0.9587** | **0.9550** |
| macro F1 | 0.766 | 0.782 |
| MAE | 0.102 | 0.108 |
| majority baseline (always "45") | 0.0429 | 0.0433 |
| text-only set (n = 1,000) | 0.996 | 1.000 |
| all-image set (n = 1,000) | 0.838 | 0.842 |
| ORACLE substitution (diagnostic, uses true labels) | 0.998 | 0.998 |
| permutation agreement / max logit difference | 1.0000 / 9.2e-5 | 1.0000 / 9.9e-5 |
| stress cases (constructed all-0 / all-9 rows) | 0 of 202 | 0 of 202 |

- **Vision is at the supervised reference.** Validation accuracy (0.9587) matches the exact-sum
  ceiling implied by a CNN with the same GELU / avg encoder trained directly on digit labels
  (0.9588 ± 0.0021 over 3 seeds, `CNN-encoder-test/`).
- **Accuracy falls with the number of image operands.** On test: 0.984 with 2 images (n = 887),
  0.953 with 5 (n = 4,899), 0.920 with 8 (n = 852), and 0.818 with 10 (n = 22).
- **Rare sums are the weak point.** That is why macro F1 is far below accuracy. Test accuracy
  is 0.963 for sums 40–49 (n = 8,225) but 0.464 for 70–79 (n = 69). Every stress case fails: rows
  summing to 0 are predicted as 12 or 14, and rows summing to 90 as 78 or 82.
- **The test split has been used** for this checkpoint. Base further development decisions on
  validation.

Evaluation outputs are in `runs/full-s0/eval_val_best/` and `runs/full-s0/eval_test_best/`. The
test evaluation is also in W&B as the run `full-s0-eval_test_best` (project
`neoscith-solo/digit-summation`); the validation evaluation was made before W&B logging existed
and is local only.

**Not run yet:**
- a sum-model run with the current default encoder (ReLU / max pooling / BatchNorm)
- more than one seed of any sum-model configuration
- augmentation, data-efficiency, and ablation comparisons
- any H100 run (none is recorded in `runs/`)

---

## Troubleshooting

| symptom | fix |
|---|---|
| `MNIST not found` / `No manifests in ...` | run `prepare_data.py --config <same config>` first |
| `Frozen data ... differs ... Refusing to overwrite` | the config changes the data; use a new `data.manifest_dir`, or `--overwrite` deliberately |
| `Manifest 'train' ... has N rows but the config asks for M` | the config and `manifest_dir` don't match; re-run `prepare_data.py` with the training config |
| `FileExistsError: runs/<name> already exists` | pick a new `logging.run_name`. Existing names include `full-s0` and `local-affine-s0`; to continue a run, use `--resume runs/<name>/checkpoints/last.pt` |
| `An operation is not implemented on MPS` | run with `--device cpu`. Or set `PYTORCH_ENABLE_MPS_FALLBACK=1` knowingly: those ops then run silently on the CPU |
| loud `PYTORCH_ENABLE_MPS_FALLBACK=1` warning | `unset PYTORCH_ENABLE_MPS_FALLBACK` |
| slow data loading on the Mac | `--set train.num_workers=4 --set train.persistent_workers=true` (speed only; the data is identical for any value) |
| a script hangs or fails with `FileNotFoundError: .../<stdin>` when using workers | macOS starts workers with "spawn", which re-imports the main script. Run code from a `.py` file with an `if __name__ == "__main__":` guard, not via `python -` / stdin |
| `train_cloud.py`: `CUDA is not available` | check `nvidia-smi`, and that torch was installed from a CUDA wheel (`torch.version.cuda` not `None`) |
| `bf16 ... not supported` | the GPU lacks bfloat16; keep `train.amp_dtype=none` |
| W&B asks for login | `wandb login`, or use `--set logging.wandb_mode=offline` / `disabled` |
| tiny overfit fails | investigate before scaling: labels (`inspect_data.py` trace), gradients (`sanity_checks.py step`), preprocessing (normalization numbers), lr |

---

## Limitations

- **One seed, previous encoder.** The only full-data result (`full-s0`) is a single seed with
  the GELU / avg / no-BatchNorm encoder. The current default encoder is untested in the sum
  model, and seed-to-seed variation is unmeasured.
- **Sum-only supervision.** The CNN must learn digit identity solely through the sum loss,
  where credit assignment over about 5 images per row is weak. With 100k sequences × 200 epochs
  this was enough to reach the supervised-reference ceiling (see Results); smaller budgets fell
  short (0.93 or less on the 20k data). An auxiliary digit loss or CNN pretraining
  (training-pool labels only, as a separately named, disabled-by-default experiment) is **not
  implemented**.
- **Classification over 0–90 cannot extrapolate.** 18 of the 91 classes never occur in the full
  training manifest, and rare sums are learned poorly: `full-s0` fails every stress case, and
  its macro F1 (0.77–0.78) is far below its accuracy.
- **The test split has been used once** (`full-s0`). Use validation for development decisions.
- **Validation is not writer-independent.** Near-duplicate images are not detected.
- **Reproducibility:** the training data stream (order + augmentation) is identical for any
  `num_workers`, persistence, or batch size, and on resume. GPU arithmetic is not bit-identical
  run to run: weights differ by ~1e-7 after 40 updates, even with an identical config.
- **Elastic distortions** are calibrated by measured displacement but are not a reproduction of
  Simard et al.

## Length-generalization bonus

**Task:** train on operand sequences of length **1–10** (uniform over lengths) and evaluate at
**every length 1–20**. The 91-way classifier can't express sums above 90, so the bonus replaces
its pooling and head with a small **autoregressive character decoder** that writes the sum as
text: `[BOS] 4 3 [EOS]`. The main-task classifier, its commands, configs, tests, and checkpoints
are unchanged. Its encoder modules moved into a shared `OperandEncoder` base class, with
parameter names and outputs verified bit-identical on `full-s0`.

**Status:** implemented and verified with bounded checks only (see
[What has been run](#bonus-what-has-been-run)). **No full bonus training run exists, so there
are no bonus results yet.**

### Data

`prepare_bonus_data.py` reuses `dataset.generate_manifest` with `sequence_length = L`:
- uniform digits;
- an image with probability `data.image_probability` (0.5), chosen independently per position;
- image operands from the split's own pool;
- targets from text integers plus MNIST labels.

Labels and sums never enter the model.

| manifests | rows (`bonus_local.yaml`) | images from |
|---|---|---|
| `train_L01` … `train_L10` | 10,000 each (100,000 total) | training pool (50,000) |
| `val_L01` … `val_L20` | 1,000 each | validation pool (10,000) |
| `test_L01` … `test_L20` | 1,000 each | official MNIST test (10,000) |
| `{val,test}_all_image_L01` … `L20` | 200 each | same pools, every operand an image |

**Images:** the pools are **the same 50k/10k/10k pools as the main task** (verified identical,
pool `9e7093ffd649107f`), so test handwriting stays held out.

**All-text overlap policy.** All-text inputs at short lengths *necessarily* repeat across splits:
length 1 has only 10 possible text inputs, and length 2 has 100 ordered ones. So the bonus does
not resample them away.
- The audit (`text_only_duplicates="report"`) still **hard-fails any shared input containing an
  image between different pools**. All-text overlaps are counted.
- In the frozen data, every all-text validation row at lengths 1–2 also occurs in training. At
  length 3 it's 79 of 114, at length 4 it's 6 of 53, and from length 5 on there are none.
- Evaluation therefore also reports **accuracy on unseen inputs**. It excludes all-text rows whose
  digits, in any order, occur in training. Lengths 11–20 are unseen by construction.
- Audit result: **486/486 hard checks passed**.

**Answers beyond training.** Training sums never exceed 90.
- Sums of 91–99 first appear at length 13.
- Sums **≥100 (three digits)** first appear at length 15 (2 of 1,000 test rows) and reach
  232 of 1,000 at length 20.

`inspect_bonus_data.py` prints this per length.

### Tokens, teacher forcing, masks, loss

| | input vocabulary (unchanged) | output vocabulary (new, saved in checkpoints) |
|---|---|---|
| special | `PAD = 0` | `PAD = 0`, `BOS = 1`, `EOS = 2` |
| digits | `"0".."9"` → 1..10 | `"0".."9"` → 3..12 |

**Operands:** each operand is still integer → string → token ID → one vector, or one CNN vector
per image. A zero operand is token 1, never padding.

**Answers:** canonical decimal, with no leading zeros except `"0"`. Teacher forcing shifts the
answer by one:

```
sum 43 -> "43" -> [BOS, "4", "3", EOS]
decoder input  = [BOS, "4", "3"]      what the decoder has seen so far
decoder target = ["4", "3", EOS]      the next character at each position
(a one-digit answer is right-padded: input [BOS, "9", PAD] -> target ["9", EOS, PAD])
```

| mask (Boolean, True = blocked/padding) | shape | where |
|---|---|---|
| `padding_mask` | [B, L] | encoder self-attention keys, decoder cross-attention memory (`memory_key_padding_mask`) |
| causal `tgt_mask` | [T, T] | decoder self-attention: position t sees only positions ≤ t |
| `decoder_padding_mask` | [B, T] | decoder self-attention keys at padded answer positions |
| target padding | [B, T] | `cross_entropy(ignore_index=PAD)` |

**Loss:** cross-entropy **averaged over valid (non-pad) target tokens in the batch**. A two-digit
answer contributes 3 tokens and a one-digit answer 2. Teacher-forced loss (`train/loss`,
`val/loss`) is never reported as accuracy.

### Model and the three variants

```
operands -> OperandEncoder.embed_operands -> [B,L,D]         one vector per operand (CNN or token embedding + modality)
         -> + sinusoidal position p for slot p                (variants with encoder positions; across the mixed sequence)
         -> bidirectional Transformer encoder (2 layers)     -> memory [B,L,D]
[BOS, d1, ...] -> answer embedding + sinusoidal positions (0 = BOS)
         -> 2 x TransformerDecoderLayer (4 heads, ff 512, pre-LN): causal self-attention + cross-attention to all of memory
         -> LayerNorm [-> + Linear(0.1 * masked sum of memory): ablation only] -> Linear -> logits [B,T,13]
```

| `bonus.architecture` | operand positions | decoder reads | parameters |
|---|---|---|---|
| `items_only_sinusoidal` (**primary**) | sinusoidal | encoder items only | 1,351,917 |
| `items_only_no_positions` | none | encoder items only | 1,351,917 |
| `sinusoidal_plus_sum_summary` (ablation) | sinusoidal | items + summary added before the head | 1,368,429 (+16,512) |

The summary layer is built last. Under the same seed, every shared parameter therefore starts
identical across variants; a test checks this.

**Sinusoidal positions.** Position p maps to `[sin(p/10000^(2i/D)), cos(p/10000^(2i/D))]`.
- This is a fixed function, not a learned table, so positions 11–20 (and output positions never
  seen in training) still get distinct, bounded vectors.
- That makes longer positions *distinguishable*. **It does not make the model count, add, or
  generalize**; those must be learned.
- Positions are computed on demand in float32, then cast to the input's dtype and device. They
  are prefix-consistent: positions 0–9 are the same whether 10 or 20 are requested.

**Why positions on operands at all, if addition is order-invariant?** It's a tradeoff:
- **Without positions:** the model is *exactly* permutation-invariant, matching the task.
  Softmax attention, however, takes weighted *averages*, so identical items can't be counted
  (see below).
- **With positions:** every slot is distinct, which makes counting possible. The cost is that the
  model is now order-sensitive, and permutation consistency has to be learned. Evaluation
  measures it.

**RoPE and ALiBi** (not implemented here) put position information inside attention:
- RoPE rotates queries and keys by position, so attention depends on relative offsets.
- ALiBi adds a distance-proportional bias to attention scores.

Both are popular for length extrapolation because they encode *relative* rather than absolute
positions. They're the natural next comparison once this simplest variant has been measured.

**The repeated-operand limitation of `items_only_no_positions`.** With identical operand vectors,
no positions, no count feature, and ordinary softmax attention, `[3]`, `[3,3]` and `[3,3,3]`
produce **identical** encoder outputs and decoder distributions. Each attention step averages
copies of the same vector.
- This was verified before any claim was made. With the actual architecture, next-token
  distributions for `[3]` vs `[3]×k` (k up to 20) agree to **3e-8** untrained, and still to
  **9e-8** after training.
- A reordered, tripled `[2,1,2,1,1,2]` also matches `[1,2]` (~5e-7).
- **More training cannot fix this.** In the tiny-overfit check, this variant fit 63 of 64 rows.
  The one it couldn't fit was `[7,7]` → 14, because the same fixed set also contained `[7]` → 7,
  which it sees as identical.
- Handwritten images of the same digit are *not* identical vectors, so image repeats are only
  near-identical. The diagnostic reports text and image repeats separately.

**The sum-summary ablation** adds `Linear(0.1 · Σ_valid memory_i)` to every decoder state after the
final LayerNorm, just before the vocabulary head.
- A sum (unlike attention's average) grows with the number of operands. Nothing normalizes it
  afterward, which would erase that.
- The 0.1 is a fixed scale, identical at every length.
- It sums *learned features*, not digit labels or the target. That makes it an additive
  architectural bias, the same one the main classifier's sum pooling provides, and not a
  programmed arithmetic solver.

### Generation

Greedy decoding starts from `BOS` and feeds the model's own previous predictions.
- The cap is `bonus.max_new_tokens = 4`: three digits plus EOS (sums up to 180). There is no
  two-digit limit.
- Completion is tracked per row; a finished row is padded after its EOS.
- There is **no special-token suppression**. An emitted PAD/BOS counts as invalid.
- Output status: `valid`, `missing_eos` (cap reached: an explicit failure), `empty`,
  `special_token`, or `leading_zero`.
- No answer is computed in code at inference.

### Training and checkpoint selection

`train_bonus_local.py` and `train_bonus_cloud.py` share `bonus_engine.run_bonus_training`, which
reuses the main engine's helpers: devices, seeding, per-(epoch, row) augmentation, optimizer,
scheduler, early stopping, resume, logging, and W&B.
- Training uses teacher forcing. Augmentation applies to training rows only.
- **Selection rule** (recorded in every checkpoint and `summary.json`): best **free-generation
  exact-answer accuracy on validation lengths 1–10**, ties broken by lower teacher-forced
  validation loss. Early stopping uses the same metric.
- **Lengths 11–20 are never evaluated during training.** Run `evaluate_bonus.py` for them, and
  don't tune repeatedly against them. Any development use of lengths 11–20 should be disclosed.
- Checkpoints store: model, optimizer, scheduler, config (architecture and positional settings),
  both vocabularies, `task = "bonus_decoder"`, data IDs, train state, and RNG.
  - Main and bonus loaders refuse each other's checkpoints.
  - Loading goes through the CPU first, so a checkpoint loads on any device.

### Evaluation (`evaluate_bonus.py --split val|test`)

Everything is free generation, reported with sample counts:
- **Per length 1–20:** exact full-answer accuracy, unseen-input accuracy, invalid and missing-EOS
  rates, and MAE on valid outputs (with that denominator). Also teacher-forced loss, plus sum
  groups **≤90 / 91–99 / ≥100**, per length and pooled over extrapolation lengths.
- **By image fraction:** results by the fraction of operands that are images, for trained vs
  extrapolation lengths.
- **Recognition vs arithmetic:**
  - **ORACLE** substitution: images replaced by their true digit strings through the normal
    tokenizer. It uses labels, and it changes the modality mix, since every row becomes
    all-text.
  - all-image sets; text-only and mixed natural rows;
  - an analytic reference `(1 − q + q·p)^L`. Here p is the CNN-encoder-test per-image accuracy
    for this encoder, and the formula is labeled as a reference, not a result.
- **Structural tests** (a correct implementation must pass, no learning involved):
  - extra right padding leaves outputs unchanged;
  - `items_only_no_positions` gives identical repeated-operand distributions;
  - the same operand gets different representations at different positions.
- **Learned tests** (correctness must be learned; these are measurements):
  - permutation consistency (shuffled operands, same sum);
  - `[d]×k` for k = 1..20, as text and as images;
  - repeated zeros → `"0"`;
  - appending a real zero (text or image) leaves the answer unchanged.
- **Galleries:** seeded examples at lengths 1/5/10/11/15/20, and a separate failure gallery.

**How to read extrapolation.** Three difficulties are mixed together, and they're distinct:
1. **Longer inputs:** more operands to aggregate, plus positions the encoder never saw.
2. **Larger sums:** 91–99 are still two digits, but the decoder never wrote them.
3. **Three-digit answers:** ≥100 needs a third digit, and an EOS at an output position never
   trained.

Sum groups condition on the total, which also changes the operands: sums ≥100 at a given length
are made of larger digits. They separate these difficulties only partly. Finally, the model
*accepting* length-20 inputs is implementation support (padding masks, positions on demand).
**Whether it generalizes is only what the per-length numbers show.**

### Commands

```bash
PY=../.torch/bin/python
# Data (frozen; re-running verifies) + inspection
$PY prepare_bonus_data.py --config configs/bonus_local.yaml
$PY inspect_bonus_data.py --config configs/bonus_local.yaml           # -> runs/inspect-bonus-local/
# Tests (bonus only, ~2 s) and the whole suite
$PY -m pytest tests/test_bonus_tokenizer.py tests/test_bonus_model.py tests/test_bonus_data.py tests/test_bonus_engine.py
$PY -m pytest
# Checks on the tiny debug data
$PY prepare_bonus_data.py --config configs/bonus_debug.yaml
$PY bonus_checks.py shapes    --config configs/bonus_debug.yaml
$PY bonus_checks.py structure --config configs/bonus_debug.yaml
$PY bonus_checks.py step      --config configs/bonus_debug.yaml
$PY bonus_checks.py overfit   --config configs/bonus_debug.yaml        # <= 300 updates; --max-steps N to change
$PY bonus_checks.py overfit   --config configs/bonus_debug.yaml --set bonus.architecture=items_only_no_positions
# Smoke run (pipeline check only)
$PY train_bonus_local.py --config configs/bonus_debug.yaml --set logging.run_name=bonus-smoke
$PY evaluate_bonus.py --checkpoint runs/bonus-smoke/checkpoints/best.pt --split val
# Primary local experiment (explicit; not run yet)
$PY train_bonus_local.py --config configs/bonus_local.yaml --set logging.run_name=bonus-sinusoidal-s0
$PY evaluate_bonus.py --checkpoint runs/bonus-sinusoidal-s0/checkpoints/best.pt --split val
# Controlled comparisons: identical data, budget, augmentation, evaluation; only the architecture changes
$PY train_bonus_local.py --config configs/bonus_local.yaml --set bonus.architecture=items_only_no_positions     --set logging.run_name=bonus-nopos-s0
$PY train_bonus_local.py --config configs/bonus_local.yaml --set bonus.architecture=sinusoidal_plus_sum_summary --set logging.run_name=bonus-sumsummary-s0
$PY compare_runs.py runs/bonus-sinusoidal-s0 runs/bonus-nopos-s0 runs/bonus-sumsummary-s0
# Final test numbers, once, for the selected checkpoints
$PY evaluate_bonus.py --checkpoint runs/bonus-sinusoidal-s0/checkpoints/best.pt --split test
# H100: copy the project with data/manifests/bonus, install CUDA torch (see Setup), then
$PY prepare_bonus_data.py --config configs/bonus_h100.yaml               # downloads MNIST, VERIFIES the copied manifests
$PY train_bonus_cloud.py  --config configs/bonus_h100.yaml --set logging.run_name=bonus-h100-sinusoidal-s0
$PY train_bonus_cloud.py  --config configs/bonus_h100.yaml --set train.amp_dtype=bf16 --set logging.run_name=bonus-h100-bf16-s0   # optional
```

**Expected time:** a rough estimate for the 100-epoch local run (39,000 updates) is about an hour
on the M4 Pro; it hasn't been measured. On debug batches, MPS model steps ran at about 4,800 seq/s
once tensor shapes had been seen, but only **462 seq/s while MPS compiles kernels for new
shapes**. Variable lengths create many shapes, so the first minutes are slow.

### Bonus: what has been run

| check | outcome |
|---|---|
| main-task regression | `full-s0` loads strictly after the `OperandEncoder` refactor, with **bit-identical logits** and identical parameter names. `evaluate.py --split val` reproduces 0.9587 accuracy and 0.7663 macro F1 |
| `pytest` | **145 passed**: 88 main task + 57 bonus (tokens and the teacher-forcing shift; positions: shape/dtype/MPS/prefix consistency; causal, operand-padding, and decoder-padding masks; loss masking and normalization; gradients for all variants; scripted generation incl. three digits / EOS / missing EOS / per-row stopping; count-blindness; position distinctness; sum-summary count sensitivity; parameter and init parity; manifests, audit policy, multi-length dataset, collate; checkpoint round trip and cross-task guards, incl. MPS load; end-to-end train + resume; validation never touches lengths > 10) |
| `prepare_bonus_data` debug + full | debug and full manifests frozen. Full: 90 manifests, 7.3 MB, pools identical to the main task, **audit 486/486** |
| `inspect_bonus_data` (debug) | answer tokens, teacher-forcing shift, and padded batch contract as shown above |
| `bonus_checks shapes` / `step` | all three variants: every boundary as diagrammed. Finite, non-zero gradients in every component; output-PAD row gradient 0; initial teacher-forced loss 2.72–2.80 (ln 13 = 2.56) |
| `bonus_checks structure` (untrained) | padding: **0** max logit difference for all variants. Repeated `[3]×k`: no-positions **3e-8**, sinusoidal 1.5e-2, sum summary 0.26. Same operand across positions: no-positions 0, sinusoidal 1.26 |
| tiny overfit (64 fixed rows, lengths 1–10, aug + dropout off, cap 300) | `items_only_sinusoidal` **passed at update 50**. `sinusoidal_plus_sum_summary` **passed at 80**. `items_only_no_positions` **63/64 after 300**: the unfit row `[7,7]`→14 is indistinguishable from `[7]`→7 in the same set, the structural limit above, not a bug |
| smoke runs (debug, 60 updates) + `evaluate_bonus --split val` | all three variants run end to end (checkpoints, curves, every report section, galleries). Accuracies of ~5–8% at lengths 1–10 are **pipeline checks, not results**. `compare_runs.py` on the three smoke runs: controlled comparison YES, the only differing setting is `bonus.architecture`. Smoke throughput was 250–1,200 seq/s (MPS shape warm-up dominates 60-update runs) |

**Not run yet:**
- any full bonus training;
- bonus test-split evaluation of a real model;
- the controlled three-variant comparison and seed repeats;
- H100 / bf16 runs.

### Bonus limitations and next steps

- **No results yet.** Every claim about extrapolation awaits the commands above.
- **Order sensitivity:** the primary model must *learn* to ignore operand order, so watch
  permutation consistency.
- **Three-digit answers** need behavior never trained: a third digit, then EOS at a new output
  position. Expect this to be the hardest group. A strict experiment never adds such training
  targets.
- **All-text overlap** at short lengths is unavoidable. Use the unseen-input columns.
- **Training speed:** shape-varying batches are slow on MPS until kernels are cached. Padding
  every batch to the maximum training length is a possible speed fix that wasn't needed for
  correctness.
- **Next comparisons** after the primary run: the two ablations (same budget), multiple seeds,
  then RoPE or ALiBi as a single additional positional method.

## References

- Y. LeCun, L. Bottou, Y. Bengio, P. Haffner. *Gradient-Based Learning Applied to Document Recognition.* Proc. IEEE 86(11), 1998. (MNIST; affine distortions)
- P. Y. Simard, D. Steinkraus, J. C. Platt. *Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis.* ICDAR 2003. (elastic distortions)
- A. Buslaev et al. *Albumentations: Fast and Flexible Image Augmentations.* Information 11(2):125, 2020.
- A. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. (sinusoidal positional encoding)
- J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* 2021. (RoPE)
- O. Press, N. A. Smith, M. Lewis. *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation.* ICLR 2022. (ALiBi)
- torchvision `datasets.MNIST`; PyTorch `nn.TransformerEncoderLayer` / `nn.TransformerDecoderLayer` / `torch.load(weights_only=True)` documentation.
