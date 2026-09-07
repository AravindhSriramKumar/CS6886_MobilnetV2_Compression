# Repository Structure — CS6886 Assignment 2

MobileNet-v2 on CIFAR-10 + a from-scratch quantization/compression pipeline.
No compression or quantization library is used anywhere (`torch.ao`,
`torch.quantization`, bitsandbytes, GPTQ, AWQ are all absent) — every scale,
rounding, clipping and bit-packing step is written by hand.

```
Assignment_2/
├── README.md                     # main writeup: Q1-Q4 with commands + result tables
├── requirements.txt              # pinned dependency versions
├── src/                          # all code (12 modules, ~1,940 lines)
├── docs/                         # answer documents for the submission PDF
├── runs/                         # checkpoints, logs, figures, result JSON/CSV
├── data/                         # CIFAR-10 (auto-downloaded; not committed)
├── wandb/                        # offline W&B runs awaiting `wandb sync`
├── Assignment_2__CS6886_-1.pdf   # the assignment brief
├── 1801.04381v4.pdf              # MobileNet-v2 paper (Sandler et al.)
└── 1911.03852v1.pdf              # HAWQ-V2 paper (Dong et al.)
```

---

## `src/` — code

Training, evaluation and compression are strictly separate stages. The compression
code imports Q1's data/model/evaluation modules and **never modifies or retrains**
the baseline checkpoint.

| File | Lines | Question | Responsibility |
|---|---|---|---|
| `data.py` | 123 | Q1a | CIFAR-10 transforms + DataLoaders. Reused unchanged by training, evaluation, calibration and Hessian estimation. Seeded generator + per-worker seeding. |
| `model.py` | 142 | Q1b | MobileNet-v2 reimplemented directly (not imported from torchvision) so individual Conv/BN layers stay addressable for Q2. CIFAR stride adaptation. Accepts a custom `inverted_residual_setting`. |
| `train.py` | 158 | Q1 | Training loop only. SGD + warmup/cosine, label smoothing; writes `best.pt`, `last.pt`, `history.csv`. |
| `evaluate.py` | 81 | Q1c | Standalone top-1 / per-class evaluation. Imported by every compression stage rather than duplicated. |
| `plot_curves.py` | 71 | Q1c | Loss/accuracy curves from `history.csv`. |
| `visualize_augmentations.py` | 104 | Q1a | Renders sample augmented images. |
| **`quantize.py`** | 202 | **Q2a** | The quantizer: `quantize_dequantize`, `weight_scales` (per-output-channel), `activation_scale`, `QuantWrapper` (fp32 / calibrate / quant modes), `wrap_model`, `calibrate`, `apply_bit_config`. |
| **`memory.py`** | 136 | **Q2c** | Manual memory accounting: `packed_bytes` (`ceil(N·b/8)` sub-byte packing), `weight_report`, `activation_report`, `per_layer_report`. Accepts a scalar bit-width or a per-layer dict. |
| **`compress.py`** | 189 | **Q2, Q4** | Compression CLI. Uniform bit-widths, or a named mixed-precision assignment via `--mixed-config`. Prints the 7 required quantities and writes JSON. |
| **`sensitivity.py`** | 121 | **Q3** | Hutchinson Hessian-trace estimation via double backprop; HAWQ-V2 scores `\|Tr(H)/n\| · ‖W−Q(W)‖²`; layer ranking. |
| **`mixed_precision.py`** | 361 | **Q3** | Bit assignment from the ranking, 21-config sweep, Pareto front, W&B logging incl. the Parallel Coordinates chart, CSV/JSON dumps. |
| `predict_compare.py` | 253 | Q3 analysis | Prediction-level comparison across precisions: flip counts vs FP32, pairwise fixed/broken/shared errors, confidence shift, per-image drill-down. |

### Module dependency flow

```
data.py ─┬─> train.py ──> runs/baseline/best.pt
         │                        │
model.py─┘                        ▼
   │            ┌──────────────────────────────────┐
   └───────────>│ quantize.py  (QuantWrapper)      │<── sensitivity.py (Q3 ranking)
                │ memory.py    (size accounting)   │
                └──────────────────────────────────┘
                     ▲                    ▲
      evaluate.py ───┴── compress.py      └── mixed_precision.py ──> W&B
                          (Q2, Q4)             (Q3 sweep)
                                                    │
                                          predict_compare.py
```

---

## `docs/` — answer documents

| File | Contents |
|---|---|
| `answers_all.md` | **Consolidated Q1–Q5 answers** — the source for the submission PDF. Opens with the summary sheet of the six figures the brief asks for on page 1. |
| `Q1_answers.md` | Q1 standalone (data pipeline, architecture, training strategy, accuracy, curves, failure modes). |
| `Q4_answers.md` | Q4 standalone (the four sub-answers for the single reported configuration). |
| `REPO_STRUCTURE.md` | This file. |

---

## `runs/` — artifacts

| Path | What it is |
|---|---|
| `baseline/best.pt` | **The Q1 checkpoint everything depends on** — 93.18% top-1, epoch 176. 8.8 MB. |
| `baseline/last.pt` | Final-epoch checkpoint (93.04%). |
| `baseline/history.csv` | Per-epoch train/test loss and accuracy, 180 rows. |
| `baseline/loss_curve.png`, `accuracy_curve.png` | Q1c figures. |
| `baseline_train.log` | Full 180-epoch training log. |
| `augmented_data_q1/*.png` | Q1a augmentation samples. |
| `compression/w<W>a<A>.json` | One file per uniform Q2 configuration (`w32a32`, `w8a8`, `w8a4`, `w4a8`, `w4a4`, `w2a8`, `w2a2`). |
| `compression/hawq_w8-4_top25_a8.json` | **The Q4 reported configuration**, with `--per-layer` breakdown. |
| `mixed_precision/results.json` | Q3 sweep: all 21 configs, Pareto front, and the full per-layer bit assignment for every config (`bit_assignments`). |
| `mixed_precision/results.csv` | Same sweep as a flat table with a `on_pareto_front` column. |
| `mixed_precision/sensitivity.json` | Per-layer Hessian traces, average traces, HAWQ scores, ranking. |
| `mixed_precision/prediction_comparison.json` | Flip statistics + indices of the 319 fixed / 166 broken images. |

`wandb/` holds 54 offline run directories (21 configs × sweep runs + summary) awaiting
`wandb sync`.

---

## Commands

```bash
# environment
conda create -n thala python=3.10 -y && conda activate thala
pip install -r requirements.txt     # torch 2.14.0, torchvision 0.29.0,
                                    # numpy 2.2.6, matplotlib 3.10.9, wandb 0.29.0

# Q1 — baseline (~48 min, one RTX A5000). Already trained; skip unless reproducing.
python src/train.py --epochs 180 --data-root ./data --output-dir ./runs/baseline
python src/evaluate.py --checkpoint runs/baseline/best.pt
python src/plot_curves.py --history runs/baseline/history.csv

# Q2 — uniform quantization; --weight-bits / --activation-bits each in {32,8,4,2}
python src/compress.py --checkpoint runs/baseline/best.pt --weight-bits 8 --activation-bits 8
python src/compress.py --checkpoint runs/baseline/best.pt --weight-bits 4 --activation-bits 4

# Q3 — Hessian sensitivity + 21-config mixed-precision sweep + W&B (~2 min)
python src/mixed_precision.py --checkpoint runs/baseline/best.pt --wandb-mode offline

# Q4 — the single reported configuration
python src/compress.py --checkpoint runs/baseline/best.pt \
    --mixed-config hawq_w8-4_top25_a8 --per-layer

# supporting analysis
python src/predict_compare.py --configs uniform_w8a8 uniform_w4a8 hawq_w8-4_top25_a8
```

Every script defaults to `--seed 42`, threaded through model init, data shuffling,
worker seeding and the Hutchinson probe vectors.

**Ordering constraint:** `mixed_precision.py` must run before
`compress.py --mixed-config ...` or `predict_compare.py`, since both read the bit
assignments from `runs/mixed_precision/results.json`.

---

## Headline results

| | Value |
|---|---|
| FP32 baseline accuracy | 93.18% top-1 |
| Reported configuration | `hawq_w8-4_top25_a8` (13/53 layers at 8-bit weights, 40 at 4-bit, activations 8-bit) |
| Quantized accuracy | 92.28% (−0.90 pt) |
| Weight compression | 7.86× (8.402 → 1.069 MB) |
| Activation compression | 4.00× (6.194 → 1.549 MB per image) |
| Model size | 8.662 MB → **1.395 MB** (6.21×) |
| Scale/metadata overhead | 68,476 B (4.7% of the compressed model) |

---

## Before publishing to GitHub

1. **Add a `.gitignore`** — `data/` is 341 MB of CIFAR-10 and must not be committed:
   ```gitignore
   data/
   wandb/
   __pycache__/
   *.pyc
   ```
   `runs/baseline/*.pt` is 8.8 MB each; keep `best.pt` (results depend on it) and
   consider dropping `last.pt`.
2. **`git init`** — the directory is not currently a git repository.
3. **`wandb login && wandb sync wandb/offline-run-*`** to publish the Parallel
   Coordinates chart and get its shareable URL for the PDF.
4. Fill the repository URL into `docs/answers_all.md` (Q5c).
