"""
Manual memory / compression accounting for CS6886 Assignment 2 (Q2).


Sub-byte packing: at 4 or 2 bits a value is not byte-addressable, so N values
of `b` bits are stored packed into ceil(N * b / 8) bytes (e.g. two 4-bit
weights per byte, four 2-bit weights per byte). Packing is done per tensor,
so at most one partially-used byte is wasted per tensor.
"""

import math

BYTES_FP32 = 4


def _bits(spec, wrapper, which):
    """Resolve a bit-width: an int applies to every layer, a dict is
    {layer_name: (w_bits, a_bits)} for Q3's mixed-precision configs."""
    if isinstance(spec, dict):
        return spec[wrapper.name][0 if which == "w" else 1]
    return spec


def packed_bytes(numel, bits):
    """Storage for `numel` values at `bits` bits each, sub-byte packed."""
    return math.ceil(numel * bits / 8)


def state_dict_elements(model):
    """Float elements in the *original* model (params + BN running stats).

    Must be called before `wrap_model`, otherwise the wrappers' own cached
    FP32/quantized weight copies would be double counted.
    """
    return sum(t.numel() for t in model.state_dict().values()
               if t.is_floating_point())


def weight_report(wrappers, total_numel, weight_bits):
    """Weight-side accounting over the whole (pre-wrap) state dict.

    Params/buffers fall into two groups:
      * Conv/Linear weights  -> quantized to `weight_bits`, packed.
      * everything else (BatchNorm affine + running stats, biases) -> FP32.
    """
    quantized_numel = sum(w.weight_numel for w in wrappers)
    other_numel = total_numel - quantized_numel

    quantized_bytes = sum(packed_bytes(w.weight_numel, _bits(weight_bits, w, "w"))
                          for w in wrappers)
    other_bytes = other_numel * BYTES_FP32

    # Metadata: one FP32 scale per output channel per layer, plus one FP32
    # activation scale per layer.
    weight_scale_count = sum(w.w_scale.numel() for w in wrappers)
    act_scale_count = len(wrappers)
    scale_bytes = (weight_scale_count + act_scale_count) * BYTES_FP32
    if weight_bits == 32:
        scale_bytes = 0  # no quantization -> no scales stored
    elif isinstance(weight_bits, dict):
        # Layers left at FP32 by a mixed-precision config need no scales.
        scale_bytes = BYTES_FP32 * sum(
            (w.w_scale.numel() + 1) for w in wrappers
            if _bits(weight_bits, w, "w") != 32)

    fp32_bytes = total_numel * BYTES_FP32
    compressed_bytes = quantized_bytes + other_bytes + scale_bytes

    return {
        "total_param_elements": total_numel,
        "quantized_weight_elements": quantized_numel,
        "fp32_other_elements": other_numel,
        "fp32_model_size_bytes": fp32_bytes,
        "fp32_model_size_mb": fp32_bytes / 2**20,
        "compressed_weight_bytes": quantized_bytes,
        "compressed_weight_mb": quantized_bytes / 2**20,
        "fp32_kept_bytes": other_bytes,
        "scale_metadata_bytes": scale_bytes,
        "num_weight_scales": weight_scale_count,
        "num_activation_scales": act_scale_count,
        "compressed_total_bytes": compressed_bytes,
        "compressed_total_mb": compressed_bytes / 2**20,
        "weight_compression_ratio": fp32_bytes / compressed_bytes,
        "weight_only_compression_ratio": (
            quantized_numel * BYTES_FP32 / max(quantized_bytes, 1)),
    }


def activation_report(wrappers, act_bits, batch_size=1):
    """Activation-side accounting: the quantized layer inputs of one forward.

    Counted per sample and scaled by `batch_size`. The FP32 baseline is the
    same tensors stored at 4 bytes/value.
    """
    per_sample = [w.in_numel for w in wrappers]
    total_elems = sum(per_sample) * batch_size
    fp32_bytes = total_elems * BYTES_FP32
    quant_bytes = sum(packed_bytes(w.in_numel * batch_size, _bits(act_bits, w, "a"))
                      for w in wrappers)
    peak_elems = max(per_sample) * batch_size if per_sample else 0

    return {
        "batch_size": batch_size,
        "activation_elements": total_elems,
        "activation_fp32_bytes": fp32_bytes,
        "activation_fp32_mb": fp32_bytes / 2**20,
        "activation_quant_bytes": quant_bytes,
        "activation_quant_mb": quant_bytes / 2**20,
        "peak_layer_activation_elements": peak_elems,
        "peak_layer_activation_quant_bytes": max(
            (packed_bytes(w.in_numel * batch_size, _bits(act_bits, w, "a"))
             for w in wrappers), default=0),
        "activation_compression_ratio": fp32_bytes / max(quant_bytes, 1),
    }


def per_layer_report(wrappers, weight_bits, act_bits):
    rows = []
    for w in wrappers:
        wb, ab = _bits(weight_bits, w, "w"), _bits(act_bits, w, "a")
        rows.append({
            "layer": w.name,
            "type": type(w.module).__name__,
            "weight_shape": list(w.w_fp32.shape),
            "weight_elements": w.weight_numel,
            "weight_fp32_bytes": w.weight_numel * BYTES_FP32,
            "weight_bits": wb,
            "activation_bits": ab,
            "weight_quant_bytes": packed_bytes(w.weight_numel, wb),
            "num_output_channel_scales": w.w_scale.numel(),
            "act_in_elements_per_sample": w.in_numel,
            "act_quant_bytes_per_sample": packed_bytes(w.in_numel, ab),
            "act_calib_absmax": float(w.act_amax),
            "act_scale": float(w.act_scale),
        })
    return rows
