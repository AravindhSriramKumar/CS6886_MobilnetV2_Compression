# CS6886 Assignment 2 — Consolidated Answers

MobileNet-v2 on CIFAR-10 + custom compression.
Repository: `src/` (code), `runs/` (checkpoints, logs, results), `docs/` (answers).

> **Status.** Q1–Q4 are complete and reproducible from the commands below. Q5 is the
> repository itself. Open items are listed at the end.

---

## Summary sheet (the figures the submission PDF asks for on page 1)

| Required item | Value |
|---|---|
| Accuracy without any compression | **93.18%** top-1 (FP32 baseline) |
| Storage overhead (metadata / scaling factors) | **68,476 B = 0.065 MB** — 17,066 per-output-channel weight scales + 53 activation scales, FP32; 4.7% of the compressed model. A further 272,936 B (0.260 MB) of BatchNorm affine + running stats and biases is deliberately kept in FP32. |
| Best compression ratio of the **model** | **6.21×** (8.662 MB → 1.395 MB) |
| Best compression ratio of the **weights** | **7.86×** (8.402 MB → 1.069 MB) |
| Best compression ratio of the **activations** | **4.00×** (6.194 MB → 1.549 MB per image) — measurement method below |
| Final approximated model size after compression | **1.395 MB** |
| W&B Parallel Coordinates chart | project `cs6886-a2-compression`, run `q3-summary` (see Q3(b)) |
| Reported configuration | `hawq_w8-4_top25_a8` — HAWQ-V2 mixed precision, **92.28%** top-1 |

**How activations were measured.** The sum of the *input* activation tensors of all 53
quantized Conv/Linear operations for **one image (batch size 1)**, taken from the tensor
shapes recorded during the calibration forward pass — i.e. exactly the tensors the
quantized operators read at inference. Intermediate BatchNorm/ReLU6 buffers and the
residual additions stay FP32 and are excluded from **both** sides of the ratio, so the
ratio is not inflated by counting FP32-only tensors in the numerator. The FP32 baseline
is the same tensor set at 4 bytes/value; the quantized side uses the same `ceil(N·b/8)`
sub-byte packing rule as the weights. Total 1,623,808 activation elements per image;
peak single layer 147,456 B.

**Environment.** `torch==2.14.0`, `torchvision==0.29.0`, `numpy==2.2.6`,
`matplotlib==3.10.9`, `wandb==0.29.0`; Python 3.10, single NVIDIA RTX A5000.
Seed `42` everywhere (model init, data shuffling, DataLoader workers, Hutchinson
probe vectors).

---
# Question 1. Training Baseline (20 points)

## (a) CIFAR-10 data preparation — normalization & augmentation (5 pts)

Code: [`src/data.py`](../src/data.py)

**Normalization.** Channel-wise mean/std computed over the full 50,000-image CIFAR-10
training split (values recomputed and printed by running the file directly):

- mean = `(0.4914, 0.4822, 0.4465)`
- std  = `(0.2470, 0.2435, 0.2616)`

These put every channel at approximately zero mean / unit variance, which keeps
BatchNorm statistics well conditioned during training.

**Transforms.**

| Split | Pipeline | Rationale |
|---|---|---|
| Train | `RandomCrop(32, padding=4, padding_mode='reflect')` | Gives the network translation invariance; reflect padding avoids the artificial black-border bias that zero-padding introduces at the crop boundary. |
| Train | `RandomHorizontalFlip(p=0.5)` | CIFAR-10 classes are left/right symmetric (a flipped cat is still a cat), so this is label-preserving. A vertical flip would not be (an upside-down truck looks wrong), so it is not used. |
| Train | `ToTensor()` → `Normalize(mean, std)` | Standard scaling into the network's expected input distribution. |
| Train (optional) | `RandomErasing(p=0.5, scale=(0.02, 0.15))` | Cutout-style occlusion regularization; exposed as a `--cutout-size` flag but left off (0) for the reported baseline run — see the failure-mode discussion in (c). |
| Test | `ToTensor()` → `Normalize(mean, std)` **only** | Deterministic — no random ops — so evaluation numbers are exactly reproducible run to run. |

**Reproducibility.** DataLoader shuffling uses a seeded `torch.Generator`, and each
worker process gets a distinct deterministic seed via `worker_init_fn`, so the
training data order is fixed for a given `--seed`.

```bash
python src/data.py     # prints the transform pipelines, recomputes the dataset
                        # stats above, and sanity-checks a batch
```

## (b) MobileNet-v2 configuration & training strategy (7 pts)

Code: [`src/model.py`](../src/model.py) (architecture), [`src/train.py`](../src/train.py) (training loop)

### Architecture

`src/model.py` reimplements the MobileNet-v2 inverted-residual architecture directly
(rather than importing `torchvision.models.mobilenet_v2`) so individual conv/BatchNorm
layers stay directly addressable for the Q2 compression pass.

CIFAR-10's 32×32 input is far smaller than ImageNet's 224×224. The stock MobileNet-v2
stride schedule downsamples by 32× total, which would collapse a 32×32 input to
below 1×1 before the classifier — so it cannot be used unmodified. Following the
standard CIFAR-MobileNet-v2 convention, two stride-2 layers are changed to stride 1:

- the stem, `Conv2d(3, 32, kernel=3, stride=1)` (ImageNet default: stride 2), and
- the first expansion stage `(t=6, c=24, n=2)`, stride 2 → 1.

All other inverted-residual stages keep their original ImageNet `(t, c, n, s)`
settings:

```
(1,  16, 1, 1)
(6,  24, 2, 1)   <- stride changed 2 -> 1 for CIFAR-10
(6,  32, 3, 2)
(6,  64, 4, 2)
(6,  96, 3, 1)
(6, 160, 3, 2)
(6, 320, 1, 1)
```

Net effect: total downsampling drops from 32× to 8×, so a 32×32 input reaches a
4×4×1280 feature map before global average pooling, instead of collapsing to ~1×1.

| Setting | Value |
|---|---|
| `width_mult` | 1.0 (no channel scaling) |
| `dropout` | 0.2 before the final linear classifier (torchvision's default) |
| BatchNorm | `eps=1e-5`, `momentum=0.1`, after every conv |
| Activation | ReLU6 throughout, matching the MobileNet-v2 paper |
| Total parameters | **2,236,682** |
| Pretraining | None — trained from scratch |

**Why train from scratch instead of using ImageNet-pretrained weights** (the
assignment explicitly allows either): transferring 224×224-pretrained weights to
32×32 CIFAR inputs would require either lossy upsampling of every image to 224×224
(expensive, and blurs the already-small objects) or the same stride surgery described
above, which invalidates most of the pretrained stem/early-block weights anyway. From
scratch training with the CIFAR-adapted stem is the standard, cheaper, more
appropriate choice at this resolution.

### Training strategy

| Setting | Value |
|---|---|
| Optimizer | SGD, momentum 0.9, Nesterov, weight_decay 5e-4 |
| LR schedule | 5-epoch linear warmup → cosine annealing to 0, base LR 0.1 |
| Batch size | 128 |
| Epochs | 180 |
| Regularization | weight decay + RandomCrop/RandomHorizontalFlip (part a) + label smoothing 0.1 |
| Seed | 42 (threaded through model init, data shuffling, and worker seeding) |

```bash
python src/train.py --epochs 180 --data-root ./data --output-dir ./runs/baseline
```

## (c) Final accuracy, curves, and failure modes (8 pts)

Trained on a single NVIDIA RTX A5000 GPU, ~16s/epoch, ~48 minutes total for 180 epochs.

**Final test top-1 accuracy: 93.18%** (best checkpoint, epoch 176 of 180).
Final-epoch (180) accuracy: 98.70% train / 93.04% test.

Per-class accuracy (best checkpoint):

| Class | Acc | Class | Acc |
|---|---|---|---|
| airplane | 93.9% | deer | 93.5% |
| automobile | 96.7% | dog | **89.8%** |
| bird | 91.3% | frog | 95.2% |
| cat | **85.6%** | horse | 94.5% |
| | | ship | 95.1% |
| | | truck | 96.2% |

**Loss curves:**

![loss curves](../runs/baseline/loss_curve.png)

**Accuracy curves:**

![accuracy curves](../runs/baseline/accuracy_curve.png)

### Failure modes

- **Class confusion.** The two weakest classes are cat (85.6%) and dog (89.8%) — the
  most visually similar pair in CIFAR-10 (similar fur texture, pose variety, and
  color distribution at 32×32 resolution). This cat/dog/bird confusion cluster is the
  standard failure mode reported for CIFAR-10 classifiers in general, not something
  specific to this architecture or training run.
- **Mid-training test-accuracy noise.** The test-accuracy curve is visibly noisy
  through the constant-high-LR middle epochs (roughly epochs 45–80), including a few
  sharp dips — an expected consequence of SGD at LR 0.1 with no LR decay yet applied.
  It smooths out and converges cleanly once the cosine schedule brings the LR near
  zero in the final ~30 epochs.
- **Mild overfitting.** Final train accuracy (98.7%) versus test accuracy (93.0%)
  leaves a ~5.7-point gap. This is expected, not a pipeline defect: a 2.2M-parameter
  model trained for 180 epochs on only 50k images with relatively light augmentation
  (crop + flip only; `--cutout-size 0` for this run, though the flag exists) will
  memorize some training-set-specific detail. A longer schedule with `RandomErasing`
  enabled (`--cutout-size > 0`) would likely narrow this gap further.

Reproduce all of the above:
```bash
python src/train.py --epochs 180 --data-root ./data --output-dir ./runs/baseline
python src/evaluate.py --checkpoint runs/baseline/best.pt --data-root ./data
python src/plot_curves.py --history runs/baseline/history.csv
```

# Question 2. Model Compression Implementation (30 points)

Code: [`src/quantize.py`](../src/quantize.py) (quantization logic),
[`src/memory.py`](../src/memory.py) (memory accounting),
[`src/compress.py`](../src/compress.py) (CLI driver).

```bash
python src/compress.py --checkpoint runs/baseline/best.pt --weight-bits 8 --activation-bits 8
python src/compress.py --checkpoint runs/baseline/best.pt --weight-bits 4 --activation-bits 4
```
Bit-widths 32 / 8 / 4 / 2 are configurable independently for weights and activations.
Results are written to `runs/compression/w<W>a<A>.json`.

## (a) The compression method and design choices (12 pts)

**Method: manual symmetric uniform quantization (static post-training quantization).**
No `torch.ao`, `torch.quantization`, bitsandbytes, GPTQ or AWQ — every scale, rounding,
clipping and packing step is written out explicitly.

```
qmax  = 2^(b-1) - 1
scale = max(|x|) / qmax
q     = clip(round(x / scale), -2^(b-1), 2^(b-1) - 1)
x_hat = q * scale
```

| Design choice | What was chosen | Why |
|---|---|---|
| Symmetric vs asymmetric | **Symmetric**, zero-point = 0 | No zero-point to store (halves the metadata) and no cross-term in the integer matmul. Post-BatchNorm weight distributions are close to zero-centred, so the wasted half-range is small. |
| Weight granularity | **Per-output-channel** | MobileNet-v2's depthwise filters have wildly different per-channel dynamic ranges; one scale per tensor lets the largest channel dictate the grid and flattens the rest to zero. Costs one FP32 scale per filter (4.7% of the compressed model — a good trade). |
| Activation granularity | **Per-tensor**, one scale per quantized layer input | Per-channel activation scales cannot be folded into a static integer pipeline without per-channel requantization at runtime. |
| Static vs dynamic | **Static** — calibrated offline, frozen at eval | Scales become a stored artifact that can be counted in the size budget; dynamic scales are transient and would make the memory accounting incoherent. |
| Calibration | running `max|x|` over 512 CIFAR-10 **train** images (8 × 64), augmentation off | Uses the train split so the test set is never touched during calibration. Deterministic (no augmentation) so ranges are reproducible. |
| Simulated arithmetic | Quantize→dequantize ("fake quantization") in FP32 | Every value a Conv/Linear sees is exactly representable on the target integer grid, so accuracy is faithful; storage cost is accounted separately at true packed sub-byte width. |
| Sub-byte storage | `ceil(N·b/8)` bytes per tensor | Two 4-bit or four 2-bit values per byte; at most one partially-used byte wasted per tensor. |

**Correctness check.** At W32/A32 the pipeline reproduces the FP32 baseline accuracy
**exactly (93.18%)**, confirming the wrapper itself introduces no error and every drop
reported below is genuine quantization damage.

## (b) How it is applied to MobileNet-v2 — layers and exceptions (9 pts)

`wrap_model` walks the module tree and replaces **every** `nn.Conv2d` and `nn.Linear`
with a `QuantWrapper` — **53 layers in total**:

| Layer group | Count | Quantized |
|---|---|---|
| Stem `ConvBNReLU(3→32)` | 1 | ✓ weights + input activations |
| Inverted residuals: pointwise expansion (1×1) | 16 | ✓ |
| Inverted residuals: depthwise (3×3, `groups=C`) | 17 | ✓ |
| Inverted residuals: pointwise-linear projection (1×1) | 17 | ✓ |
| Final `ConvBNReLU(320→1280, 1×1)` | 1 | ✓ |
| Classifier `Linear(1280→10)` | 1 | ✓ weights + input activations |

**Exceptions (deliberately kept in FP32):**

- **BatchNorm** — affine parameters and running mean/var. They are few (68,234 elements
  total, 0.26 MB) but directly rescale every activation; quantizing them corrupts the
  distribution the activation scales were calibrated for, for almost no size gain.
- **Biases** — negligible in count, and bias error adds directly to the output.
- **Final logits** — the classifier's *output* is left FP32; only its weights and input
  activation are quantized, so the argmax is computed at full precision.
- **Residual additions, ReLU6, pooling** — untouched; they carry no parameters.

## (c) Storage overheads, included in the size estimate (9 pts)

Every number below is computed by hand in `src/memory.py` from tensor shapes and the
chosen bit-widths.

For the reported configuration (`hawq_w8-4_top25_a8`, see Q4):

| Component | Elements | Bytes | MB | Share |
|---|---|---|---|---|
| Quantized Conv/Linear weights (packed) | 2,202,560 | 1,121,432 | 1.069 | 76.7% |
| FP32 kept: BatchNorm affine + running stats, biases | 68,234 | 272,936 | 0.260 | 18.7% |
| **Metadata: 17,066 weight scales + 53 activation scales (FP32)** | 17,119 | **68,476** | **0.065** | **4.7%** |
| **Total compressed model** | | **1,462,844** | **1.395** | 100% |
| FP32 original | 2,270,794 | 9,083,176 | 8.662 | |

The metadata is the price of per-output-channel granularity: 17,066 scales = one per
output filter across all 53 layers. It is why the whole-model ratio (6.21×) is lower
than the weights-only ratio (7.86×), and it is included in every size and ratio quoted
in this report.

## Q2 results — uniform bit-widths

| W/A bits | Top-1 | Compressed weights | Weight ratio *(weights only)* | Activation mem (batch 1) | Act ratio |
|---|---|---|---|---|---|
| 32/32 | 93.18% | 8.402 MB | 1.00× | 6.194 MB | 1.00× |
| 8/8 | 92.78% | 2.101 MB | 4.00× | 1.549 MB | 4.00× |
| 4/8 | 90.75% | 1.050 MB | 8.00× | 1.549 MB | 4.00× |
| 8/4 | 22.36% | 2.101 MB | 4.00× | 0.774 MB | 8.00× |
| 4/4 | 21.83% | 1.050 MB | 8.00× | 0.774 MB | 8.00× |
| 2/8 | 12.70% | 0.525 MB | 16.00× | 1.549 MB | 4.00× |
| 2/2 | 10.00% | 0.525 MB | 16.00× | 0.387 MB | 16.00× |

**The key finding, which drives Q3:** weights tolerate 4 bits (W4A8 = 90.75%) but
activations do not (W8A4 = 22.36%). The collapse is entirely on the activation side.
MobileNet-v2's depthwise and ReLU6 activations have long-tailed per-tensor ranges, so a
single max-abs scale spends most of its 16 levels on rare outliers, quantizes the bulk
of the distribution to near zero, and the error compounds across 53 layers.

---

# Question 3. Compression Results

Code: [`src/sensitivity.py`](../src/sensitivity.py),
[`src/mixed_precision.py`](../src/mixed_precision.py).

```bash
python src/mixed_precision.py --checkpoint runs/baseline/best.pt --wandb-mode offline
```
Outputs `runs/mixed_precision/results.{json,csv}` and `sensitivity.json`.

## (a) Compression applied at different levels

Two axes were swept: **uniform** bit-widths (Q2), and **HAWQ-V2 sensitivity-guided
mixed precision**, 21 configurations in total.

**Sensitivity metric.** For each of the 53 Conv/Linear layers, the Hessian trace of the
loss w.r.t. that layer's weights is estimated by Hutchinson's method — for a Rademacher
vector `v`, `E[vᵀHv] = Tr(H)`, with `Hv` obtained by double backprop through the
gradient (8 batches × 8 probe draws, plain autograd, no library). The HAWQ-V2 score is

```
Omega_l = |Tr(H_l) / n_l| · ||W_l - Q_b(W_l)||²      (b = 4)
```

i.e. curvature × the perturbation 4-bit quantization would actually inflict. The
absolute value is deliberate: the trained model sits near but not exactly at a minimum
and the estimator is stochastic, so a few layers return small negative trace estimates;
using the signed value would rank those as the *safest* to crush to 2 bits, which is
backwards.

The top-10 most sensitive layers are all in the **early** network
(`features.1.conv.0.0`, `features.2.conv.1.0`, `features.2.conv.2`, `features.4.conv.2`,
`features.1.conv.1`, `features.7.conv.2`, `features.0.0`, …) — early layers operate at
the highest spatial resolution and their errors propagate through every later block.

**Bit assignment.** The top-`frac` of the ranking receives 8-bit weights, the remainder
4-bit or 2-bit. Activations either stay at 8 bits (`_a8`) or follow the same split
(`_amix`).

## (b) Accuracy comparison + W&B Parallel Coordinates chart

| Config | avg W bits | avg A bits | Top-1 | Model ratio *(incl. FP32 BN + scales)* | Act ratio | Model (MB) | Total ratio | Pareto |
|---|---|---|---|---|---|---|---|---|
| `uniform_w32a32` | 32.00 | 32.00 | 93.18% | 1.00× | 1.00× | 8.662 | 1.00× | ✓ |
| `uniform_w8a8` | 8.00 | 8.00 | 92.78% | 3.57× | 4.00× | 2.426 | 3.74× |  |
| `uniform_w4a8` | 4.00 | 8.00 | 90.75% | 6.30× | 4.00× | 1.376 | 5.08× | ✓ |
| `uniform_w4a4` | 4.00 | 4.00 | 21.83% | 6.30× | 8.00× | 1.376 | 6.91× | ✓ |
| `uniform_w2a8` | 2.00 | 8.00 | 12.70% | 10.18× | 4.00× | 0.851 | 6.19× |  |
| `hawq_w8-4_top10_a8` | 4.38 | 8.00 | 91.56% | 6.28× | 4.00× | 1.380 | 5.07× | ✓ |
| `hawq_w8-4_top10_amix` | 4.38 | 4.38 | 45.84% | 6.28× | 6.76× | 1.380 | 6.47× | ✓ |
| `hawq_w8-4_top25_a8` | 4.98 | 8.00 | 92.28% | 6.21× | 4.00× | 1.395 | 5.05× | ✓ |
| `hawq_w8-4_top25_amix` | 4.98 | 4.98 | 64.85% | 6.21× | 5.75× | 1.395 | 6.01× | ✓ |
| `hawq_w8-4_top50_a8` | 5.96 | 8.00 | 92.69% | 5.02× | 4.00× | 1.724 | 4.54× | ✓ |
| `hawq_w8-4_top50_amix` | 5.96 | 5.96 | 83.68% | 5.02× | 4.70× | 1.724 | 4.89× |  |
| `hawq_w8-4_top75_a8` | 7.02 | 8.00 | 92.94% | 4.26× | 4.00× | 2.036 | 4.14× | ✓ |
| `hawq_w8-4_top75_amix` | 7.02 | 7.02 | 87.43% | 4.26× | 4.30× | 2.036 | 4.28× |  |
| `hawq_w8-2_top10_a8` | 2.57 | 8.00 | 10.12% | 10.11× | 4.00× | 0.857 | 6.18× |  |
| `hawq_w8-2_top10_amix` | 2.57 | 2.57 | 10.00% | 10.11× | 10.31× | 0.857 | 10.19× | ✓ |
| `hawq_w8-2_top25_a8` | 3.47 | 8.00 | 10.26% | 9.85× | 4.00× | 0.880 | 6.12× |  |
| `hawq_w8-2_top25_amix` | 3.47 | 3.47 | 10.00% | 9.85× | 7.37× | 0.880 | 8.64× |  |
| `hawq_w8-2_top50_a8` | 4.94 | 8.00 | 65.75% | 6.31× | 4.00× | 1.373 | 5.09× | ✓ |
| `hawq_w8-2_top50_amix` | 4.94 | 4.94 | 10.00% | 6.31× | 5.16× | 1.373 | 5.77× |  |
| `hawq_w8-2_top75_a8` | 6.53 | 8.00 | 88.48% | 4.71× | 4.00× | 1.841 | 4.38× |  |
| `hawq_w8-2_top75_amix` | 6.53 | 6.53 | 10.00% | 4.71× | 4.47× | 1.841 | 4.61× |  |

*(`avg W/A bits` are the mean bit-widths across the 53 layers; the total ratio combines
model and per-image activation memory. Full table also in `runs/mixed_precision/results.csv`.)*

**Pareto-optimal set** (non-dominated on accuracy vs total compression): `uniform_w32a32`,
`hawq_w8-4_top75_a8`, `hawq_w8-4_top50_a8`, `hawq_w8-4_top25_a8`, `hawq_w8-4_top10_a8`,
`uniform_w4a8`, `hawq_w8-2_top50_a8`, `hawq_w8-4_top25_amix`, `hawq_w8-4_top10_amix`,
`uniform_w4a4`, `hawq_w8-2_top10_amix`.

### What the sweep shows

1. **Mixed precision strictly beats uniform 8-bit.** `hawq_w8-4_top75_a8` reaches
   **92.94%** vs uniform 8-bit's 92.78% while being **16% smaller** (2.036 vs 2.426 MB) —
   better on *both* axes. The sensitivity ranking correctly identifies which quarter of
   the network genuinely needs 8 bits.
2. **At matched size, HAWQ beats uniform by +1.53 pt.** `hawq_w8-4_top25_a8` = 92.28%
   vs `uniform_w4a8` = 90.75%, at 1.395 vs 1.376 MB.
3. **2-bit weights are unusable here.** Accuracy only recovers to 88.48% at `top75`,
   where three quarters of layers are back at 8 bits and the compression benefit is
   gone. MobileNet-v2's depthwise filters have too few weights per channel to survive a
   4-level grid.
4. **Every mixed-*activation* variant is dominated**, confirming the Q2 finding: the
   binding constraint is activation dynamic range, not bit allocation. No sensitivity
   ranking can rescue max-abs scaling at 4 bits — that needs percentile/MSE clipping.

### Prediction-level verification

Code: [`src/predict_compare.py`](../src/predict_compare.py). A single-image comparison
would not be evidence (any two models disagree somewhere), so the full disagreement
structure over all 10,000 test images was measured:

| Config | Top-1 | Predictions changed vs FP32 | correct→wrong | wrong→correct | Agreement with FP32 |
|---|---|---|---|---|---|
| `uniform_w8a8` | 92.78% | 154 | 87 | 47 | 98.5% |
| `uniform_w4a8` | 90.75% | 614 | 389 | 146 | 93.9% |
| `hawq_w8-4_top25_a8` | 92.28% | 334 | 186 | 96 | 96.7% |

Quantization flips predictions in **both** directions — even W4A8 corrects 146 images the
FP32 model gets wrong — so net accuracy is a small difference of two large numbers.
Mean confidence on unflipped images barely moves (p(true) 0.831 → 0.832; margin 3.765 →
3.710): damage is concentrated in a few hundred already-marginal images rather than
spread as a uniform confidence sag.

Head-to-head, `hawq_w8-4_top25_a8` vs `uniform_w4a8`: 925 → 772 errors, **606 shared**,
**319 fixed / 166 broken** (net +153 images = +1.53 pt). The mixed config genuinely
*fixes* errors rather than relocating them.

Representative images (drawn from the flip lists, not cherry-picked):

| Image | True | FP32 | `uniform_w4a8` | `hawq_w8-4_top25_a8` |
|---|---|---|---|---|
| #12 | dog | dog (p=0.781) | **cat** (p=0.313, margin 0.148) | dog (p=0.750) |
| #15 | ship | ship (p=0.880) | **frog** (p=0.847) — confidently wrong | ship (p=0.757) |
| #49 | frog | frog (p=0.672) | frog (p=0.379) | **bird** (p=0.775) — a case the mixed config breaks |

### W&B Parallel Coordinates chart

All 21 configurations are logged to W&B, project **`cs6886-a2-compression`**, group
`q3-mixed-precision` — one run per configuration plus a `q3-summary` run carrying the
required **Parallel Coordinates** plot with axes:

```
avg_weight_bits → avg_activation_bits → frac_high → weight_compression_ratio
→ activation_compression_ratio → compressed_model_mb → total_compression_ratio
→ test_accuracy
```

The summary run also carries an accuracy-vs-compression scatter and the ranked
per-layer sensitivity bar chart.

> **Action required before submission:** runs are currently logged in **offline** mode
> under `./wandb/`. Run `wandb login`, then `wandb sync wandb/offline-run-*` (or re-run
> with `--wandb-mode online`) to obtain the shareable chart URL, and paste the chart
> screenshot here.

---

# Question 4. Compression Analysis

**Single reported configuration: `hawq_w8-4_top25_a8`** — HAWQ-V2 mixed precision, the
13 most Hessian-sensitive of the 53 Conv/Linear layers at 8-bit weights, the remaining
40 at 4-bit weights, all activations at 8 bits.

```bash
python src/compress.py --checkpoint runs/baseline/best.pt \
    --mixed-config hawq_w8-4_top25_a8 --per-layer
```

## (a) Compression ratio of the weights — **7.86×**

| Quantity | Value |
|---|---|
| Quantized weight elements (Conv/Linear) | 2,202,560 |
| As FP32 | 8,810,240 B = 8.402 MB |
| Packed at the assigned 8/4 bits | 1,121,432 B = 1.069 MB |
| Effective average width | 4.07 bits/weight |
| **Weight compression ratio** | **7.86×** |

## (b) Compression ratio of the activations — **4.00×**

Measurement method as stated in the summary sheet: sum of the input activation tensors
of all 53 quantized Conv/Linear ops at batch size 1, from shapes recorded during
calibration; FP32 baseline is the same tensor set at 4 B/value; FP32-only intermediates
excluded from both sides.

| Quantity | Value |
|---|---|
| Activation elements per image | 1,623,808 |
| As FP32 | 6,495,232 B = 6.194 MB |
| At 8 bits | 1,623,808 B = 1.549 MB |
| Peak single-layer activation | 147,456 B |
| **Activation compression ratio** | **4.00×** |

## (c) Accuracy at that compression — **92.28% top-1**

| | Top-1 |
|---|---|
| FP32 baseline | 93.18% |
| `hawq_w8-4_top25_a8` | **92.28%** |
| Drop | **−0.90 pt** |

## (d) Final approximated model size — **1.395 MB** (from 8.662 MB, **6.21×**)

| Component | Bytes | MB |
|---|---|---|
| Quantized Conv/Linear weights (8/4-bit, packed) | 1,121,432 | 1.069 |
| FP32 kept: BatchNorm affine + running stats, biases | 272,936 | 0.260 |
| Scale metadata (17,066 weight + 53 activation scales, FP32) | 68,476 | 0.065 |
| **Total compressed model** | **1,462,844** | **1.395** |
| FP32 original | 9,083,176 | 8.662 |

### Why this configuration

It is the knee of the Pareto front. Configurations with more precision cost 25–46% more
memory for ≤0.7 pt of accuracy; everything more compressed falls off a cliff (uniform
W4A4 → 21.83%, every 2-bit weight configuration → ≤65.75% unless three quarters of the
layers are restored to 8 bits). It also **dominates uniform 8-bit on size** (1.395 vs
2.426 MB at −0.50 pt) and **dominates uniform W4A8 on accuracy** (+1.53 pt at the same
size).

---

# Question 5. Reproducibility & Repository (10 points)

## (a) Modular codebase — separation of training / evaluation / compression

| File | Responsibility |
|---|---|
| `src/data.py` | CIFAR-10 transforms and DataLoaders (Q1a) — reused unchanged by training, evaluation, and calibration |
| `src/model.py` | MobileNet-v2 architecture (Q1b) |
| `src/train.py` | Training loop only (Q1) |
| `src/evaluate.py` | Standalone evaluation — reused by every compression stage |
| `src/quantize.py` | Manual quantization primitives and the layer wrapper (Q2) |
| `src/memory.py` | Memory / compression accounting (Q2c) |
| `src/compress.py` | Compression CLI — uniform (Q2) and mixed (Q4) |
| `src/sensitivity.py` | Hutchinson Hessian traces + HAWQ-V2 scores (Q3) |
| `src/mixed_precision.py` | Mixed-precision sweep, Pareto, W&B logging (Q3) |
| `src/predict_compare.py` | Prediction-level comparison across precisions |
| `src/plot_curves.py`, `src/visualize_augmentations.py` | Figures |

Training, evaluation and compression are strictly separate stages: the compression code
imports Q1's data/model/evaluation modules and **never modifies or retrains** the
baseline checkpoint. Q3 extended Q2's modules only additively (`QuantWrapper.set_bits`,
`apply_bit_config`, per-layer bit dicts in the accounting), and `src/compress.py` still
reproduces its Q2 numbers exactly.

## (b) Exact commands, environment, dependency versions, seeds

```bash
conda create -n thala python=3.10 -y && conda activate thala
pip install -r requirements.txt      # torch==2.14.0, torchvision==0.29.0,
                                     # numpy==2.2.6, matplotlib==3.10.9, wandb==0.29.0

# Q1 — train the baseline (~48 min on one RTX A5000)
python src/train.py --epochs 180 --data-root ./data --output-dir ./runs/baseline
python src/evaluate.py --checkpoint runs/baseline/best.pt
python src/plot_curves.py --history runs/baseline/history.csv

# Q2 — uniform quantization at any bit-width (32 / 8 / 4 / 2)
python src/compress.py --checkpoint runs/baseline/best.pt --weight-bits 8 --activation-bits 8
python src/compress.py --checkpoint runs/baseline/best.pt --weight-bits 4 --activation-bits 4

# Q3 — sensitivity analysis + mixed-precision sweep + W&B
python src/mixed_precision.py --checkpoint runs/baseline/best.pt --wandb-mode offline

# Q4 — the single reported configuration
python src/compress.py --checkpoint runs/baseline/best.pt \
    --mixed-config hawq_w8-4_top25_a8 --per-layer

# Supporting analysis
python src/predict_compare.py --configs uniform_w8a8 uniform_w4a8 hawq_w8-4_top25_a8
```

**Seed configuration.** `--seed 42` (default) is threaded through model initialization,
DataLoader shuffling (a seeded `torch.Generator`), per-worker seeding
(`worker_init_fn`), and the Hutchinson probe-vector generator. Test-time transforms are
deterministic (no augmentation), and calibration uses a fixed, unshuffled prefix of the
train split, so all reported accuracy and memory numbers reproduce exactly.

## (c) GitHub repository link

> **To fill in:** `https://github.com/<user>/<repo>`

---

# Open items before submission

1. **W&B runs are offline.** Run `wandb login` && `wandb sync wandb/offline-run-*` to get
   the shareable Parallel Coordinates chart URL + screenshot for the PDF.
2. **GitHub link** not yet filled in (Q5c); repository is not currently a git repo.
3. **Optional accuracy headroom.** A reference CIFAR MobileNet-v2 implementation
   (`chenhang98/mobileNet-v2_cifar10`) reaches ~94.5% with an *identical* parameter count
   (2,236,682) by downsampling only 4× (final feature map 8×8) instead of 8× (4×4), plus
   300 epochs. Adopting that stride schedule is a one-line change but requires retraining
   (~1.7 h at 34.6 s/epoch) and re-running Q2–Q4 (~5 min). It would lift every accuracy
   figure by roughly 1 pt; model size and weight-compression ratios are unchanged.
4. **Optional compression headroom.** The activation ratio is capped at 4.00× because
   4-bit activations collapse under max-abs calibration. Percentile clipping (calibrate
   to the 99.9th percentile of |x| rather than the max) is a small change to
   `src/quantize.py` and would plausibly make 4-bit activations usable, roughly doubling
   the activation ratio. Not implemented.
