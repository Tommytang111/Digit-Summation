# CNN-encoder-test: how well can the sum model's vision encoder recognize digits?

The digit-summation model learns digit identity **only through the sum loss**. Its CNN image
encoder never sees a digit label. This folder trains a reference: the **same encoder
architecture** with a 10-way linear head, the **same training augmentation**, and the **same
training images**, but supervised directly with MNIST digit labels. The reference shows the
digit accuracy and macro F1 this encoder design can reach on the official MNIST test set when it
is told the answer.

## What is held identical to the sum model

| | sum model | this reference |
|---|---|---|
| encoder | `model.ImageEncoder` (conv 3×3 32 → GELU → avgpool 2 → conv 3×3 64 → GELU → avgpool 2 → Linear 3136→128 → LayerNorm) | the same class, built from `configs/local.yaml` (a test checks the parameter shapes match) |
| augmentation | `build_transform(cfg.augment, train=True)`: affine (rotate ±10°, translate ±2 px, scale 0.9–1.1, shear ±5°), black fill, then Normalize | the same call, seeded per (epoch, image) |
| training images | the 50,000-image training pool | the same pool (`data/manifests/full`, pool `9e7093ffd649107f`) |
| model selection | validation pool (10,000 images) | best epoch by validation digit accuracy |
| test | official MNIST test (10,000 images) | the same images, evaluated **once** with the selected weights |
| optimizer | AdamW lr 3e-4, wd 0.01, warmup 500 + cosine, grad clip 1.0 | the same settings (`engine.build_optimizer` / `build_scheduler`) |

What differs: supervision (digit labels instead of sums), the 10-way head instead of the
Transformer + 91-way head, batch 128, and 30 epochs (390 updates per epoch, 11,700 total).

## The implied sum ceiling

Each validation and test image is predicted once. The script then takes the sum model's **own**
sequence sets (`val`, `test`, `val_all_image`, `test_all_image` from `data/manifests/full`),
reads every image operand with this CNN, keeps text operands as they are, and adds:

> **implied sum ceiling** = the exact-sum accuracy if digit recognition were exactly this good
> and addition were perfect.

During development, compare your sum model's **validation** accuracy against the **val**
ceiling. Keep test for final reporting.
- **Near the ceiling:** the encoder reads digits about as well as supervised training allows.
- **Well below it:** either the encoder or the aggregation is losing accuracy.

The report also gives the independent-errors approximation p^k (p = per-image accuracy,
k = number of images in the row). The true ceiling is usually a little higher, because two wrong
digits can still sum correctly.

## Run it

```bash
cd /Users/tommy/Projects/Digit-Summation
../.torch/bin/python -m pytest CNN-encoder-test                                   # 5 synthetic tests, CPU
../.torch/bin/python CNN-encoder-test/digit_cnn.py --run-name relu-max-bn-s0      # the config's encoder (ReLU/max/BN)
../.torch/bin/python CNN-encoder-test/digit_cnn.py --activation gelu --pooling avg --no-batch-norm --run-name digit-cnn-s0  # original encoder
../.torch/bin/python CNN-encoder-test/digit_cnn.py --epochs 1 --max-steps 30 --device cpu --run-name smoke
```

Options: `--epochs`, `--batch-size`, `--max-steps`, `--device`, `--config`, and `--set key=value`
(for example `--set augment.mode=none` or `--set model.cnn_pooling=max`). Avoid running it while
another job trains on the GPU, because both slow down.

Outputs go to `CNN-encoder-test/runs/<run-name>/`:

| file | contents |
|---|---|
| `metrics.json` | per-epoch history; the best epoch; val and test reports (accuracy, macro F1, per-digit precision/recall/F1/support, confusion matrix); implied sum ceilings |
| `best.pt` | weights of the validation-selected epoch + config |
| `test_predictions.csv` | one row per test image: label, prediction, confidence |
| `curves.png` | loss / accuracy / macro F1 vs optimizer step (train = augmented epoch, val = eval mode) |
| `confusion_test.png` | test confusion matrix (errors colored, diagonal = correct counts) |
| `misclassified_test.png` | a seeded random draw of test errors with true → predicted labels |

## Results

Run `digit-cnn-s0` (2026-10-15, M4 Pro MPS, single seed): 30 epochs, 11,700 updates, 51 s of
training. The best epoch, 29 (step 11,310), was chosen on validation.

| digit classification | accuracy | macro F1 | errors |
|---|---|---|---|
| validation (10,000 images) | 0.9920 | 0.9919 | 80 |
| **test (10,000 official MNIST test images)** | **0.9929** | **0.9929** | **71** |

Test per-digit F1 ranges from 0.9887 to 0.9964: 0: 0.9964, 1: 0.9951, 2: 0.9932, 3: 0.9926,
4: 0.9959, 5: 0.9922, 6: 0.9927, 7: 0.9922, 8: 0.9887, 9: 0.9896. The most common confusions
are 5→3 (5), 2→7 (4), 8→3 (4), and 9→8 (4). See `confusion_test.png`.

Implied sum ceilings (this CNN reads every image, addition is perfect):

| sequences | exact-sum accuracy | sum macro F1 | p^k approximation |
|---|---|---|---|
| `val` (20,000) | 0.9611 | 0.9601 | 0.9606 |
| `val_all_image` (1,000) | 0.9160 | 0.8835 | 0.9228 |
| `test` (20,000) | 0.9638 | 0.9417 | 0.9650 |
| `test_all_image` (1,000) | 0.9380 | 0.9435 | 0.9312 |

**Compared with the sum model (`full-s0`, best checkpoint at step 69k, validation only).** Using the
3-seed baseline mean below (val ceiling **0.9588 ± 0.0021**), not the single seed above:
- **Accuracy:** `full-s0`'s val exact-sum accuracy is **0.9587**, essentially at the baseline
  CNN's ceiling. An analysis of `full-s0`'s errors by image count implied about 99.2% per-image
  accuracy, the same as this CNN's ~99.15–99.20% validation digit accuracy. **The encoder reads
  digits about as well as this supervised reference.**
- **Macro F1:** val macro F1 is **0.766**, against ~0.95–0.96 if digits were read at this CNN's
  accuracy and added perfectly. That gap is not vision. It comes from rarely seen sum classes:
  the sum model fails on sums with fewer than ~100 training rows even when given perfect digits
  (F1 0.34 on those classes).

The single-seed table above varies by seed (see below). Differences of a few tenths of a point are within run-to-run and
sampling noise (20,000 val sequences share 10,000 images).

## Limitations

- **Single seed:** seed-to-seed variation is not measured.
- **Reference, not an upper bound:** this is one supervised recipe. A longer or tuned schedule
  could do slightly better.
- **The ceiling is a diagnostic, not a model.** It uses this CNN's digit reads plus perfect
  arithmetic.
- **Test-set use:** this experiment evaluates on the official test images, which the sum model's
  test sequences also use. Base development decisions on the validation numbers.

## Variant: ReLU + max pooling + BatchNorm (3 seeds each)

The variant changes three things at once. ReLU replaces GELU, max pooling replaces average
pooling, and BatchNorm follows each conv (conv biases removed). Run it with
`--activation relu --pooling max --batch-norm`; it builds `VariantImageEncoder`, with the same
layer layout and +96 parameters. Everything else is identical to the baseline: augmentation,
images, AdamW lr 3e-4, warmup 500 + cosine, 30 epochs, batch 128. Seeds are 0, 1, and 2 for
both; baseline seed 0 is `digit-cnn-s0`. Aggregate with `compare_cnn_runs.py`.

| mean ± std over 3 seeds | baseline (GELU / avg / no BN) | variant (ReLU / max / BN) | difference |
|---|---|---|---|
| val digit accuracy | 0.9915 ± 0.0005 | 0.9924 ± 0.0002 | +0.09 pts |
| test digit accuracy | 0.9923 ± 0.0006 | 0.9929 ± 0.0009 | +0.06 pts |
| test errors (of 10,000) | 77.3 ± 5.5 | 70.7 ± 8.7 | about 7 fewer |
| val implied sum ceiling | 0.9588 ± 0.0021 | 0.9629 ± 0.0004 | +0.41 pts |
| best epoch | 28 ± 1 | 25 ± 4 | |

How to read it:
- **Validation favors the variant consistently.** Its worst seed (0.9923) beats the baseline's
  best (0.9920). The difference is about 3 standard errors, though with only 3 seeds.
- **Test points the same way** (+0.06 points), but the gap is within about 1 standard error.
- Per-image gains compound over the ~5 images per sequence. That makes the implied sum ceiling
  +0.4 points, the number that matters for the sum model.
- All three changes moved together, so the gain can't be attributed to any one of them.
- This was measured on standalone digit classification. Whether it transfers to the sum model,
  which is trained only through the sum loss and where BatchNorm sees the images gathered from
  all slots of a batch, is untested.

**Adopted (2026-10-15):** the sum model's `ImageEncoder` now defaults to ReLU + max pooling +
BatchNorm. `digit_cnn.py` builds the encoder from the config's `model` section, so it keeps
mirroring the sum model; reproduce the original with `--activation gelu --pooling avg --no-batch-norm`.
The variant runs above used a separate copy of this layout. Their weights load into the new
`ImageEncoder` and give identical outputs (checked).
