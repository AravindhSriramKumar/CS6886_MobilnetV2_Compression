# CS6886 — Systems for Deep Learning · Assignment 2
### MobileNet-v2 on CIFAR-10 

Train MobileNet-v2 from scratch on CIFAR-10, then compress it with a quantization
scheme written from scratch (no `torch.ao`, `torch.quantization`, bitsandbytes, GPTQ
or AWQ), and report weight/activation compression ratios, accuracy and final model
size.

**Headline result:** 93.18% FP32 top-1 → **92.28% top-1 at 1.395 MB**
(from 8.662 MB): **7.86× weight**, **4.00× activation**, **6.21× whole-model**
compression, for a **0.90 pt accuracy drop** (config `hawq_w8-4_top25_a8`).
---

## 1. Requirements

| | |
|---|---|
| Python | 3.10 |
| GPU | any CUDA GPU with ≥8 GB (developed on an NVIDIA RTX A5000); CPU works but training is impractically slow |
| Disk | ~1 GB (CIFAR-10 ≈ 170 MB + checkpoints) |
| Training time | ~16 s/epoch → ~48 min for the full 180-epoch baseline on one A5000 |

Pinned dependencies live in [`requirements.txt`](requirements.txt):
`torch==2.14.0`, `torchvision==0.29.0`, `numpy==2.2.6`, `matplotlib==3.10.9`,
`wandb==0.29.0`.

## 2. Setup

```bash
git clone https://github.com/AravindhSriramKumar/CS6886_MobilnetV2_Compression.git
cd CS6886_MobilnetV2_Compression

conda create -n cs6886a2 python=3.10 -y
conda activate cs6886a2
# or: python3.10 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If the pinned `torch`/`torchvision` wheels are not available for your CUDA version,
install the matching build from https://pytorch.org first, then
`pip install -r requirements.txt` for the remaining packages.

**Data.** CIFAR-10 is *not* committed (see `.gitignore`). It is downloaded
automatically into `./data` by `torchvision.datasets.CIFAR10` the first time any
script runs. **Run every command from the repository root**, so `--data-root ./data`
resolves to the same copy and the dataset is downloaded only once.

W&B is optional. Every script defaults to `--wandb-mode offline` (runs land in
`./wandb/`); run `wandb login` once and pass `--wandb-mode online` to stream live,
or push offline runs later with `wandb sync wandb/offline-run-*`.

## 3. Repository layout

```
├── src/                          # all source code
│   ├── data.py                   # Q1a: CIFAR-10 transforms, seeded loaders, dataset stats
│   ├── model.py                  # Q1b: MobileNet-v2 (CIFAR stride schedule), 2,236,682 params
│   ├── train.py                  # Q1b: training loop (SGD + warmup + cosine)
│   ├── evaluate.py               # Q1c: top-1 / per-class accuracy from a checkpoint
│   ├── plot_curves.py            # Q1c: loss & accuracy curves from history.csv
│   ├── visualize_augmentations.py# Q1a: augmentation sample grid
│   ├── quantize.py               # Q2: hand-written symmetric uniform quantizer
│   ├── memory.py                 # Q2/Q4: weight + activation memory accounting
│   ├── compress.py               # Q2/Q4: CLI driver (uniform or mixed-precision)
│   ├── sensitivity.py            # Q3: Hutchinson Hessian traces + HAWQ-V2 scores
│   ├── mixed_precision.py        # Q3: 21-config sweep, Pareto front, W&B logging
│   └── predict_compare.py        # Q3/Q4: per-image prediction diff between configs
├── runs/                         # committed results
│   ├── baseline/best.pt          # trained FP32 checkpoint (93.18%) — reuse to skip training
│   ├── baseline/history.csv      # per-epoch train/test loss & accuracy
│   ├── compression/*.json        # Q2 uniform W/A results
│   └── mixed_precision/          # Q3 sweep: results.{json,csv}, sensitivity.json
├── docs/                         # written answers (Q1–Q4)
├── requirements.txt
└── README.md
```

## 4. Reproducing everything

Every script is seeded (`--seed 42`, default) and takes explicit `--data-root` /
`--output-dir` paths, so the pipeline reproduces on any machine meeting section 1.

```bash
# --- Q1: data pipeline, training, evaluation -------------------------------
python src/data.py                                     # transforms + dataset stats sanity check
python src/visualize_augmentations.py                  # -> runs/augmented_data_q1/
python src/train.py --epochs 180 --data-root ./data --output-dir ./runs/baseline
python src/evaluate.py --checkpoint runs/baseline/best.pt --data-root ./data
python src/plot_curves.py --history runs/baseline/history.csv

# --- Q2: uniform post-training quantization --------------------------------
for W in 32 8 4 2; do for A in 32 8 4 2; do
  python src/compress.py --checkpoint runs/baseline/best.pt \
      --weight-bits $W --activation-bits $A
done; done                                             # -> runs/compression/w<W>a<A>.json

# --- Q3: HAWQ-V2 sensitivity + mixed-precision sweep -----------------------
python src/mixed_precision.py --checkpoint runs/baseline/best.pt --wandb-mode offline
                                                       # -> runs/mixed_precision/{results.json,results.csv,sensitivity.json}

# --- Q4: the single reported configuration ---------------------------------
python src/compress.py --checkpoint runs/baseline/best.pt \
    --mixed-config hawq_w8-4_top25_a8 --per-layer
python src/predict_compare.py --checkpoint runs/baseline/best.pt \
    --reference uniform_w4a8 --candidate hawq_w8-4_top25_a8

# --- Report figures & PDF ---------------------------------------------------
python report/make_figures.py                          # -> report/figures/
cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

`runs/baseline/best.pt` is committed, so **Q2–Q4 can be reproduced in minutes
without retraining** — skip the `train.py` line and run the rest as-is. Retraining
from scratch with the same seed and the same PyTorch/CUDA build reproduces the
baseline; small (<0.2 pt) deviations across different GPU architectures or cuDNN
versions are expected from non-deterministic kernels.

Every script supports `--help` for the full flag list (bit-widths, calibration
batches, batch size, workers, device, output paths).

## 5. Written answers

- Full write-up: [`docs/answers_all.md`](docs/answers_all.md)
The remainder of this README is the technical write-up, question by question.

---

## Q1(a) — Data pipeline
Code: `src/data.py`

```bash
python src/data.py     # prints transforms, recomputes dataset stats, sanity-checks a batch
```

Normalization constants (channel-wise, computed over the 50,000 CIFAR-10 train images):
- mean = (0.4914, 0.4822, 0.4465)
- std  = (0.2470, 0.2435, 0.2616)

| Split | Transform pipeline | Why |
|---|---|---|
| Train | `RandomCrop(32, padding=4, padding_mode='reflect')` | translation invariance; reflect padding avoids the black-border bias of zero padding |
| Train | `RandomHorizontalFlip(p=0.5)` | CIFAR-10 classes are left/right symmetric → label preserving (vertical flip is not) |
| Train | `ToTensor()` → `Normalize(mean, std)` | ~zero-mean/unit-variance per channel; keeps BatchNorm statistics well conditioned |
| Train | *(optional)* `RandomErasing(p=0.5, scale=(0.02,0.15))` | Cutout-style occlusion regularization for longer schedules |
| Test  | `ToTensor()` → `Normalize(mean, std)` **only** | fully deterministic, so eval numbers are reproducible |

Reproducibility: seeded `torch.Generator` for shuffling + `worker_init_fn` giving each
DataLoader worker a distinct deterministic seed.

## Q1(b) — Model & training strategy

Code: `src/model.py` (architecture), `src/train.py` (training loop)

```bash
python src/train.py --epochs 180 --data-root ./data --output-dir ./runs/baseline
```

**MobileNet-v2 configuration.** `src/model.py` reimplements the inverted-residual
architecture directly (rather than importing `torchvision.models.mobilenet_v2`) so
individual conv/BN layers stay directly addressable for the Q2 compression pass.
CIFAR-10's 32x32 input is much smaller than ImageNet's 224x224, so the stock stride
schedule (32x total downsampling) would collapse the feature map to sub-1px before
the classifier. Following the standard CIFAR-MobileNet-v2 convention, two stride-2
layers are changed to stride 1:
- the stem `Conv2d(3, 32, k=3, stride=1)` (ImageNet default: stride 2), and
- the first expansion stage `(t=6, c=24, n=2)`, stride 2 → 1.

All other stages keep their original `(t, c, n, s)` settings: `(1,16,1,1) (6,24,2,1)
(6,32,3,2) (6,64,4,2) (6,96,3,1) (6,160,3,2) (6,320,1,1)`. Net effect: 8x total
downsampling instead of 32x, so a 32x32 input reaches a 4x4 feature map (1280
channels) before global average pooling — versus collapsing to ~1x1 with the
unmodified schedule. Other settings: `width_mult=1.0` (no channel scaling),
`dropout=0.2` before the final linear classifier (torchvision's default), BatchNorm
with `eps=1e-5, momentum=0.1` after every conv, ReLU6 activations throughout —
all matching the MobileNet-v2 paper. Total parameters: **2,236,682**. Trained from
scratch (no ImageNet pretraining), since transferring 224x224 pretrained weights to
32x32 inputs would require lossy upsampling and stride adjustments that undo most of
the benefit of pretraining.

**Training strategy.**

| Setting | Value |
|---|---|
| Optimizer | SGD, momentum 0.9, Nesterov, weight_decay 5e-4 |
| LR schedule | 5-epoch linear warmup → cosine annealing to 0, base LR 0.1 |
| Batch size | 128 |
| Epochs | 180 |
| Regularization | weight decay + RandomCrop/RandomHorizontalFlip (Q1a) + label smoothing 0.1 |
| Seed | 42 (`--seed`, threaded through model init, shuffling, and worker seeding) |

Reproducibility: `set_seed()` seeds `random`, `numpy`, and `torch`
(CPU + all CUDA devices) before model construction and dataloader creation.

## Q1(c) — Baseline results

Ran on a single NVIDIA RTX A5000 (~16s/epoch, ~48 min total for 180 epochs).

- **Final test top-1 accuracy: 93.18%** (best checkpoint, epoch 176 of 180)
- Final-epoch (180) accuracy: 98.70% train / 93.04% test
- Per-class accuracy (best checkpoint): airplane 93.9%, automobile 96.7%, bird 91.3%,
  **cat 85.6%**, deer 93.5%, **dog 89.8%**, frog 95.2%, horse 94.5%, ship 95.1%,
  truck 96.2%

Loss curves: `runs/baseline/loss_curve.png` — Accuracy curves: `runs/baseline/accuracy_curve.png`

Reproduce the reported numbers:
```bash
python src/train.py --epochs 180 --data-root ./data --output-dir ./runs/baseline
python src/evaluate.py --checkpoint runs/baseline/best.pt --data-root ./data
python src/plot_curves.py --history runs/baseline/history.csv
```

**Failure modes / discussion.** The two weakest classes are cat (85.6%) and dog
(89.8%) — the most visually similar pair in CIFAR-10 (similar fur texture, pose, and
color distribution), and this bird/cat/dog confusion cluster is the standard failure
mode reported for CIFAR-10 classifiers generally, not specific to this architecture.
The accuracy curve shows the expected pattern for this recipe: a noisy test-accuracy
trace through the constant-high-LR middle epochs (SGD with LR 0.1 causes visible test
accuracy oscillation, e.g. brief dips around epochs 45-80), which smooths out and
converges cleanly once the cosine schedule brings the LR near zero in the last ~30
epochs. Final train accuracy (98.7%) versus test accuracy (93.0%) shows a ~5.7pt gap,
i.e. mild overfitting consistent with a 2.2M-parameter model trained for 180 epochs
on only 50k images with light augmentation — expected and not a pipeline defect,
since the augmentation (crop+flip) is intentionally light per Q1a's design (no Cutout
used for this run: `--cutout-size 0`).

## Q2 — Manual quantization (weights + activations)
Code: `src/quantize.py` (quantization logic), `src/memory.py` (memory/compression
accounting), `src/compress.py` (CLI driver). Q1 code (`data.py`, `model.py`,
`evaluate.py`) is reused unchanged — the baseline is **not** retrained.

```bash
python src/compress.py --checkpoint runs/baseline/best.pt --weight-bits 4 --activation-bits 4
```
Results are written to `runs/compression/w<W>a<A>.json` (`--per-layer` adds a
per-layer breakdown). Both bit-widths accept 32 / 8 / 4 / 2.

**Scheme.** Symmetric uniform quantization, zero-point 0, written by hand (no
`torch.ao` / `torch.quantization` / bitsandbytes / GPTQ / AWQ):
`scale = max|x| / (2^(b-1)-1)`, `q = clip(round(x/scale), -2^(b-1), 2^(b-1)-1)`,
`x̂ = q·scale`.
- Weights: **per-output-channel** scales, computed statically from the trained FP32
  weights of every Conv/Linear — stem conv, all 17 inverted residuals (pointwise
  expansion, depthwise, pointwise-linear projection), the final 1×1 conv and the
  classifier (53 layers total).
- Activations: one scale per layer input, calibrated as running max|x| over a small
  CIFAR-10 train subset (default 8×64 = 512 images), then **frozen** for evaluation.
- BatchNorm (affine + running stats), biases and the final logits stay FP32.
- Sub-byte storage is accounted manually: N values at b bits cost `ceil(N·b/8)` bytes
  (two 4-bit or four 2-bit values per byte), plus FP32 scale metadata (one per output
  channel per layer + one activation scale per layer, ~66.9 KB).

| W/A bits | Test top-1 | Compressed weights | Weight ratio (model / weights only) | Activation mem (batch 1) | Act. ratio |
|---|---|---|---|---|---|
| 32/32 | 93.18% | 8.402 MB | 1.00× / 1.00× | 6.194 MB | 1.00× |
| 8/8   | 92.78% | 2.101 MB | 3.57× / 4.00× | 1.549 MB | 4.00× |
| 8/4   | 22.36% | 2.101 MB | 3.57× / 4.00× | 0.774 MB | 8.00× |
| 4/4   | 21.83% | 1.050 MB | 6.30× / 8.00× | 0.774 MB | 8.00× |
| 2/8   | 12.70% | 0.525 MB | 10.18× / 16.00× | 1.549 MB | 4.00× |
| 2/2   | 10.00% | 0.525 MB | 10.18× / 16.00× | 0.387 MB | 16.00× |

FP32 model size is 8.662 MB (2,270,346 float elements incl. BN running stats); the
"model" ratio is against that total and is capped by the 266.5 KB of FP32 BN/bias
tensors that are deliberately not quantized. W32/A32 reproduces the FP32 accuracy
exactly (93.18%), confirming the wrapper itself is lossless.

**Discussion.** Plain post-training max-abs quantization holds up at 8 bits (−0.40 pt)
but collapses below that. Weights alone at 4 bits are not the whole story — the drop
appears as soon as *activations* go to 4 bits (8/4 is already 22.4%): MobileNet-v2's
depthwise and ReLU6 activations have long-tailed per-tensor ranges, so a single
max-abs scale spends most of its 16 levels on rare outliers and quantizes the bulk of
the distribution to near zero, and the error compounds across 53 layers. Recovering
low-bit accuracy needs percentile/MSE clipping, per-channel activation handling or
quantization-aware fine-tuning — deliberately left to Q3.

## Q3 — HAWQ-V2 sensitivity-guided mixed precision
Code: `src/sensitivity.py` (Hutchinson Hessian traces + HAWQ-V2 scores),
`src/mixed_precision.py` (sweep / Pareto / W&B driver). Q1 is untouched; Q2's
`quantize.py` / `memory.py` gained only backward-compatible extensions
(`QuantWrapper.set_bits`, `apply_bit_config`, and per-layer bit dicts in the
memory accounting) — `src/compress.py` still reproduces its Q2 numbers exactly.

```bash
wandb login                       # optional; omit and everything runs offline
python src/mixed_precision.py --checkpoint runs/baseline/best.pt --wandb-mode offline
```
Outputs: `runs/mixed_precision/results.{json,csv}` (one row per config, Pareto flag)
and `runs/mixed_precision/sensitivity.json` (per-layer traces, scores, ranking).

**Sensitivity.** For each of the 53 Conv/Linear layers, the Hessian trace of the
loss w.r.t. that layer's weights is estimated with Hutchinson's method —
`E[vᵀHv] = Tr(H)` for Rademacher `v`, with `Hv` obtained by double backprop
(8 batches × 8 draws, plain autograd, no library). The HAWQ-V2 score is
`Ω_l = |Tr(H_l)/n_l| · ‖W_l − Q_b(W_l)‖²`: curvature × the perturbation 4-bit
quantization would actually inflict. The absolute value is deliberate — the
trained model is near but not exactly at a minimum, so a few layers return small
negative trace estimates, and using the signed value would rank those as the
*safest* to crush, which is backwards. The top-`frac` of the ranking gets 8 bits,
the rest 4 or 2. Activations either stay at 8 bits (`fixed`) or follow the same
split (`mixed`).

Selected results from the 21-config sweep (full table in the CSV):

| Config | Top-1 | Weight ratio | Act. ratio | Model size | Total ratio | Pareto |
|---|---|---|---|---|---|---|
| uniform_w32a32 | 93.18% | 1.00× | 1.00× | 8.662 MB | 1.00× | ✓ |
| uniform_w8a8 | 92.78% | 3.57× | 4.00× | 2.426 MB | 3.74× | |
| **hawq_w8-4_top75_a8** | **92.94%** | 4.26× | 4.00× | 2.036 MB | 4.14× | ✓ |
| hawq_w8-4_top50_a8 | 92.69% | 5.02× | 4.00× | 1.724 MB | 4.54× | ✓ |
| **hawq_w8-4_top25_a8** | **92.28%** | 6.21× | 4.00× | **1.395 MB** | **5.05×** | ✓ |
| hawq_w8-4_top10_a8 | 91.56% | 6.28× | 4.00× | 1.380 MB | 5.07× | ✓ |
| uniform_w4a8 | 90.75% | 6.30× | 4.00× | 1.376 MB | 5.08× | ✓ |
| hawq_w8-2_top75_a8 | 88.48% | 4.71× | 4.00× | 1.841 MB | 4.03× | |
| hawq_w8-2_top50_a8 | 65.75% | 6.31× | 4.00× | 1.373 MB | 5.09× | ✓ |
| uniform_w4a4 | 21.83% | 6.30× | 8.00× | 1.376 MB | 6.91× | ✓ |

**Reading the sweep.**
- Mixed precision buys real accuracy at nearly free cost. `hawq_w8-4_top75_a8`
  beats uniform 8-bit (92.94% vs 92.78%) while being **16% smaller** (2.036 vs
  2.426 MB) — the sensitivity ranking correctly identifies which quarter of the
  network genuinely needs 8 bits.
- At matched size, HAWQ beats uniform: `hawq_w8-4_top25_a8` is +1.53 pt over
  `uniform_w4a8` (92.28% vs 90.75%) at 1.395 vs 1.376 MB.
- 2-bit weights are not usable here even with protection: only at `top75`
  (three quarters kept at 8 bits, so the compression benefit is gone) does
  accuracy recover to 88.5%. MobileNet-v2's depthwise filters have too few
  weights per channel to survive a 4-level grid.
- Mixed-*activation* variants are all dominated. This confirms the Q2 finding:
  the binding constraint is activation range, not weight precision. Sub-8-bit
  activations need percentile/MSE clipping rather than a bit-allocation fix, and
  no sensitivity ranking can rescue max-abs scaling at 4 bits.

**Recommended for Q4: `hawq_w8-4_top25_a8`** — 92.28% top-1 (−0.90 pt vs FP32),
1.395 MB compressed model (from 8.662 MB), 6.21× weight compression, 4.00×
activation compression, 5.05× total. It is the knee of the Pareto front: the
configs above it cost 25–46% more memory for ≤0.7 pt, and everything more
compressed than it falls off a cliff. If accuracy is the priority instead,
`hawq_w8-4_top75_a8` is the only config that beats uniform-8-bit on *both* axes.
The exact per-layer bit assignment for every config is stored under
`bit_assignments` in `results.json`.

**W&B.** Each of the 21 configs is logged as its own run (group `q3-mixed-precision`,
project `cs6886-a2-compression`) plus a `q3-summary` run carrying the required
**Parallel Coordinates** plot over `avg_weight_bits → avg_activation_bits →
frac_high → weight_compression_ratio → activation_compression_ratio →
compressed_model_mb → total_compression_ratio → test_accuracy`, an
accuracy-vs-compression scatter, and the ranked layer-sensitivity bar chart.
Runs default to `--wandb-mode offline` (they land in `./wandb/` and can be pushed
later with `wandb sync wandb/offline-run-*`); run `wandb login` once and pass
`--wandb-mode online` to log live.

## Q4 — Compression analysis (single reported configuration)
Answers: [`docs/Q4_answers.md`](docs/Q4_answers.md). The assignment asks for **one**
configuration, not a ratio-vs-accuracy table; the Q3 sweep is the justification for
the choice and Q4 reports only `hawq_w8-4_top25_a8`.

```bash
python src/compress.py --checkpoint runs/baseline/best.pt \
    --mixed-config hawq_w8-4_top25_a8 --per-layer
```
(`--mixed-config NAME` loads a per-layer bit assignment from
`runs/mixed_precision/results.json`; without it `compress.py` behaves exactly as in Q2.)

| Q4 | Answer |
|---|---|
| (a) weight compression ratio | **7.86×** (2,202,560 weights: 8.402 MB FP32 → 1.069 MB packed, avg 4.07 bits) |
| (b) activation compression ratio | **4.00×** (6.194 MB → 1.549 MB per image) |
| (c) accuracy | **92.28%** top-1 (FP32 93.18%, −0.90 pt) |
| (d) final model size | **1.395 MB** (from 8.662 MB; whole-model **6.21×**) |

**How activations were measured:** the sum of the *input* activation tensors of all 53
quantized Conv/Linear ops for one image (batch size 1), from shapes recorded during the
calibration pass — the tensors the quantized operators actually read. FP32 baseline is
the same tensor set at 4 B/value; intermediate BN/ReLU6 buffers and residual adds are
FP32 and excluded from *both* sides, so the ratio is not inflated.

Storage overhead is included in the size: 68,476 B of scales (17,066 per-output-channel
weight scales + 53 activation scales) = 4.7% of the compressed model, plus 272,936 B of
deliberately-unquantized FP32 BatchNorm/bias tensors = 18.7%.
