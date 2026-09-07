# Q4 — Compression Analysis

**Single reported configuration: `hawq_w8-4_top25_a8`**
(HAWQ-V2 sensitivity-guided mixed precision: the 13 most Hessian-sensitive of the
53 Conv/Linear layers at 8-bit weights, the remaining 40 at 4-bit weights;
all activations 8-bit. Per-output-channel weight scales, calibrated per-tensor
activation scales, BatchNorm/biases/logits kept FP32.)

Reproduce with one command:

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
| **Weight compression ratio** | **7.86×** |

Sub-byte packing is accounted manually: `ceil(N·b/8)` bytes per tensor, so two
4-bit weights share a byte. Effective average width is 4.07 bits/weight.

## (b) Compression ratio of the activations — **4.00×**

**How measured.** Activation memory is the sum of the *input* activation tensors
of all 53 quantized Conv/Linear ops for **one image (batch size 1)**, counted
from the shapes recorded during the calibration forward pass — i.e. the tensors
the quantized operators actually read at inference, not intermediate BN/ReLU6
buffers or the residual adds (those stay FP32 and are excluded from both sides
of the ratio, so the ratio is not inflated). FP32 baseline is the same tensor
set at 4 bytes/value; the quantized side uses the same `ceil(N·b/8)` packing.

| Quantity | Value |
|---|---|
| Activation elements per image | 1,623,808 |
| As FP32 | 6,495,232 B = 6.194 MB |
| At 8 bits | 1,623,808 B = 1.549 MB |
| Peak single-layer activation | 147,456 B (`features.3/4` 24×64×64 stage) |
| **Activation compression ratio** | **4.00×** |

## (c) Accuracy at that compression — **92.28% top-1**

| | Top-1 |
|---|---|
| FP32 baseline (Q1) | 93.18% |
| `hawq_w8-4_top25_a8` | **92.28%** |
| Drop | −0.90 pt |

Prediction-level check (`src/predict_compare.py`): 96.7% of test predictions are
unchanged vs FP32; 186 images flip correct→wrong and 96 flip wrong→correct.
Against uniform W4A8 at the same size, this config fixes 319 images and breaks
166 — a genuine +1.53 pt, not relocated error.

## (d) Final approximated model size — **1.395 MB** (from 8.662 MB)

| Component | Bytes | MB |
|---|---|---|
| Quantized Conv/Linear weights (8/4-bit, packed) | 1,121,432 | 1.069 |
| FP32 kept: BatchNorm affine + running stats, biases (68,234 elems) | 272,936 | 0.260 |
| Scale/metadata: 17,066 per-output-channel weight scales + 53 activation scales (FP32) | 68,476 | 0.065 |
| **Total compressed model** | **1,462,844** | **1.395** |
| FP32 original (2,270,794 float elements) | 9,083,176 | 8.662 |
| **Whole-model compression ratio** | | **6.21×** |

Storage overhead is fully included above: metadata is 4.7% of the compressed
model, and the deliberately-unquantized FP32 BN/bias tensors are 18.7% — the two
together are why the whole-model ratio (6.21×) is below the weight-only ratio
(7.86×).

## Why this configuration

It is the knee of the Q3 Pareto front. Configs with more precision cost 25–46%
more memory for ≤0.7 pt of accuracy; everything more compressed falls off a
cliff (uniform W4A4 → 21.8%, any 2-bit weight config → ≤65.8%). It also
dominates uniform 8-bit on size (1.395 vs 2.426 MB) at −0.50 pt, and dominates
uniform W4A8 on accuracy (+1.53 pt) at the same size.
