"""
Manual symmetric uniform quantization for CS6886 Assignment 2 (Q2).

No torch.ao / torch.quantization / bitsandbytes / GPTQ / AWQ -- every scale,
rounding and clipping step below is written out explicitly.

Scheme (symmetric, uniform, zero-point = 0):

    qmax  = 2^(bits-1) - 1
    scale = max(|x|) / qmax
    q     = clip(round(x / scale), -qmax-1, qmax)
    x_hat = q * scale

* Weights: per-output-channel scales (one scale per Conv2d/Linear output
  filter), computed statically from the trained FP32 weights.
* Activations: one scale per quantized layer input, calibrated as the running
  max |x| over a small CIFAR-10 calibration subset, then *frozen* for eval.
* BatchNorm parameters, biases and the final logits stay FP32.

Inference is simulated ("fake quantization"): tensors are quantized to the
integer grid and immediately dequantized, so the arithmetic runs in FP32 but
every value seen by a Conv/Linear is exactly representable at the target
bit-width. Storage cost is accounted separately in `memory.py` using packed
sub-byte widths.
"""

import torch
import torch.nn as nn

SUPPORTED_BITS = (32, 8, 4, 2)


def qmax_for(bits):
    """Positive end of the symmetric integer grid, e.g. 127 for 8 bits."""
    return 2 ** (bits - 1) - 1


def quantize_dequantize(x, scale, bits):
    """Symmetric uniform quantize -> clip -> dequantize (fake quantization)."""
    if bits == 32:
        return x
    qmax = qmax_for(bits)
    scale = torch.clamp(scale, min=1e-12)
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return q * scale


def weight_scales(weight, bits):
    """Per-output-channel scales: max|W| over each output filter / qmax."""
    if bits == 32:
        return torch.ones(weight.shape[0], device=weight.device)
    flat = weight.detach().reshape(weight.shape[0], -1)
    amax = flat.abs().amax(dim=1)
    # A dead (all-zero) filter would give scale 0 -> guard it.
    return torch.clamp(amax, min=1e-12) / qmax_for(bits)


def activation_scale(amax, bits):
    """Single tensor-wide activation scale from a calibrated max|x|."""
    if bits == 32:
        return torch.ones((), device=amax.device)
    return torch.clamp(amax, min=1e-12) / qmax_for(bits)


class QuantWrapper(nn.Module):
    """Wraps one Conv2d/Linear: quantizes its input activation and weights.

    Modes:
      'fp32'      -- passthrough (used to sanity-check the wrapper is lossless)
      'calibrate' -- records running max|input|, otherwise passthrough
      'quant'     -- fake-quantizes input (frozen scale) and weights
    """

    def __init__(self, module, weight_bits, act_bits, name=""):
        super().__init__()
        assert isinstance(module, (nn.Conv2d, nn.Linear))
        self.module = module
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.name = name
        self.mode = "fp32"

        self.register_buffer("act_amax", torch.zeros(()))
        self.register_buffer("act_scale", torch.ones(()))
        # Shape of one sample's input/output activation, filled during calibration.
        self.in_numel = 0
        self.out_numel = 0

        w = module.weight
        self.register_buffer("w_scale", weight_scales(w, weight_bits))
        self.register_buffer("w_fp32", w.detach().clone())
        # Per-output-channel scale broadcast over the remaining weight dims.
        view = [-1] + [1] * (w.dim() - 1)
        self.register_buffer(
            "w_quant", quantize_dequantize(w.detach(), self.w_scale.view(view),
                                           weight_bits))

    @property
    def weight_numel(self):
        return self.w_fp32.numel()

    def freeze_activation_scale(self):
        self.act_scale = activation_scale(self.act_amax, self.act_bits)

    def set_bits(self, weight_bits=None, act_bits=None):
        """Re-target this layer to new bit-widths (Q3 mixed precision).

        Recomputes the per-output-channel weight scales and the quantized
        weight copy in place, and re-derives the activation scale from the
        already-calibrated max|x| (which is bit-width independent), so a
        sweep over many mixed-precision configs needs only one calibration
        pass over the data.
        """
        if weight_bits is not None and weight_bits != self.weight_bits:
            self.weight_bits = weight_bits
            self.w_scale = weight_scales(self.w_fp32, weight_bits)
            view = [-1] + [1] * (self.w_fp32.dim() - 1)
            self.w_quant = quantize_dequantize(
                self.w_fp32, self.w_scale.view(view), weight_bits)
        if act_bits is not None:
            self.act_bits = act_bits
            self.freeze_activation_scale()

    def forward(self, x):
        if self.mode == "calibrate":
            self.act_amax = torch.maximum(self.act_amax,
                                          x.detach().abs().amax().float())
            self.in_numel = x[0].numel()
            out = self._run(x, self.w_fp32)
            self.out_numel = out[0].numel()
            return out
        if self.mode == "quant":
            xq = quantize_dequantize(x, self.act_scale, self.act_bits)
            return self._run(xq, self.w_quant)
        return self._run(x, self.w_fp32)

    def _run(self, x, weight):
        m = self.module
        if isinstance(m, nn.Conv2d):
            return nn.functional.conv2d(x, weight, m.bias, m.stride,
                                        m.padding, m.dilation, m.groups)
        return nn.functional.linear(x, weight, m.bias)


def wrap_model(model, weight_bits, act_bits, bit_config=None):
    """Recursively replace every Conv2d/Linear with a QuantWrapper.

    This covers the whole MobileNet-v2 graph: the stem conv, every inverted
    residual (pointwise expansion, depthwise, pointwise-linear projection),
    the final 1x1 conv and the classifier Linear. BatchNorm/ReLU6/pooling and
    the residual adds are left untouched in FP32.

    `bit_config` optionally overrides the uniform `weight_bits`/`act_bits` per
    layer, as {layer_name: (w_bits, a_bits)} -- this is what Q3's mixed
    precision uses. Layers absent from it keep the uniform defaults.
    """
    wrappers = []
    bit_config = bit_config or {}

    def _recurse(parent, prefix):
        for child_name, child in list(parent.named_children()):
            full = f"{prefix}{child_name}"
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                wb, ab = bit_config.get(full, (weight_bits, act_bits))
                w = QuantWrapper(child, wb, ab, name=full)
                setattr(parent, child_name, w)
                wrappers.append(w)
            else:
                _recurse(child, full + ".")

    _recurse(model, "")
    return wrappers


def set_mode(wrappers, mode):
    for w in wrappers:
        w.mode = mode


def apply_bit_config(wrappers, bit_config):
    """Re-target an already-calibrated wrapped model to {name: (w_bits, a_bits)}."""
    for w in wrappers:
        if w.name in bit_config:
            wb, ab = bit_config[w.name]
            w.set_bits(wb, ab)


@torch.no_grad()
def calibrate(model, wrappers, loader, device, num_batches):
    """Run `num_batches` calibration batches to collect activation ranges."""
    model.eval()
    set_mode(wrappers, "calibrate")
    seen = 0
    for i, (x, _) in enumerate(loader):
        if i >= num_batches:
            break
        model(x.to(device))
        seen += x.size(0)
    for w in wrappers:
        w.freeze_activation_scale()
    set_mode(wrappers, "quant")
    return seen
