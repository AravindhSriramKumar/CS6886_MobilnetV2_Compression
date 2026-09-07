"""
Q2 compression driver for CS6886 Assignment 2.

    Q1 FP32 checkpoint -> manual symmetric quantization of every Conv/Linear
    (per-output-channel weight scales + calibrated activation scales)
    -> inference -> accuracy + memory/compression report (JSON).

Q1 code is reused as-is: `data.get_dataloaders`, `model.build_model`,
`evaluate.evaluate`. Nothing in Q1 is modified or retrained.

    python src/compress.py --checkpoint runs/baseline/best.pt \
        --weight-bits 4 --activation-bits 4
"""

import argparse
import json
import os
import time

import torch

from data import get_dataloaders
from evaluate import evaluate
from memory import (activation_report, per_layer_report,
                    state_dict_elements, weight_report)
from model import build_model
from quantize import SUPPORTED_BITS, calibrate, set_mode, wrap_model


def load_mixed_config(path, name):
    """Fetch one named per-layer bit assignment produced by Q3's sweep."""
    with open(path) as f:
        assignments = json.load(f)["bit_assignments"]
    if name not in assignments:
        raise SystemExit(f"unknown config '{name}'. available: "
                         f"{', '.join(assignments)}")
    return {n: tuple(v) for n, v in assignments[name].items()}


def load_baseline(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    targs = ckpt.get("args", {})
    model = build_model(num_classes=10,
                        width_mult=targs.get("width_mult", 1.0),
                        dropout=targs.get("dropout", 0.2)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="runs/baseline/best.pt")
    p.add_argument("--weight-bits", type=int, default=8, choices=SUPPORTED_BITS)
    p.add_argument("--activation-bits", type=int, default=8,
                   choices=SUPPORTED_BITS)
    p.add_argument("--mixed-config", type=str, default=None,
                   help="name of a Q3 per-layer bit assignment (e.g. "
                        "hawq_w8-4_top25_a8); overrides --weight-bits/"
                        "--activation-bits with the mixed-precision recipe")
    p.add_argument("--assignments", type=str,
                   default="runs/mixed_precision/results.json")
    p.add_argument("--calib-batches", type=int, default=8,
                   help="calibration batches drawn from the CIFAR-10 train split")
    p.add_argument("--calib-batch-size", type=int, default=64)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--mem-batch-size", type=int, default=1,
                   help="batch size used for the activation-memory accounting")
    p.add_argument("--output-dir", type=str, default="runs/compression")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--skip-fp32-eval", action="store_true")
    p.add_argument("--per-layer", action="store_true",
                   help="also dump the per-layer breakdown into the JSON")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, ckpt = load_baseline(args.checkpoint, device)

    # Calibration set: a small subset of the *train* split with augmentation
    # off (deterministic ranges); evaluation uses the untouched test split.
    calib_loader, _ = get_dataloaders(root=args.data_root,
                                      batch_size=args.calib_batch_size,
                                      num_workers=args.num_workers,
                                      augment=False, seed=args.seed)
    _, test_loader = get_dataloaders(root=args.data_root,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     augment=False, seed=args.seed)

    fp32_acc = None
    if not args.skip_fp32_eval:
        _, fp32_acc, _ = evaluate(model, test_loader, device)
        print(f"FP32 baseline test accuracy: {fp32_acc*100:.2f}%")

    bit_config = None
    if args.mixed_config:
        bit_config = load_mixed_config(args.assignments, args.mixed_config)

    baseline_numel = state_dict_elements(model)
    wrappers = wrap_model(model, args.weight_bits, args.activation_bits,
                          bit_config)
    model.to(device)
    if bit_config:
        wb = sorted({b for b, _ in bit_config.values()})
        ab = sorted({a for _, a in bit_config.values()})
        print(f"wrapped {len(wrappers)} Conv/Linear layers "
              f"(mixed '{args.mixed_config}': weights {wb} bits, "
              f"activations {ab} bits)")
    else:
        print(f"wrapped {len(wrappers)} Conv/Linear layers "
              f"(W{args.weight_bits}A{args.activation_bits})")

    n_calib = calibrate(model, wrappers, calib_loader, device,
                        args.calib_batches)
    print(f"calibrated activation ranges on {n_calib} images")

    set_mode(wrappers, "quant")
    t0 = time.time()
    q_loss, q_acc, per_class = evaluate(model, test_loader, device)
    print(f"quantized test accuracy: {q_acc*100:.2f}%  "
          f"(loss {q_loss:.4f}, {time.time()-t0:.1f}s)")

    w_spec = bit_config if bit_config else args.weight_bits
    a_spec = bit_config if bit_config else args.activation_bits
    w_rep = weight_report(wrappers, baseline_numel, w_spec)
    a_rep = activation_report(wrappers, a_spec, batch_size=args.mem_batch_size)

    results = {
        "config": {
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": ckpt.get("epoch"),
            "mixed_config": args.mixed_config,
            "weight_bits": (sorted({b for b, _ in bit_config.values()})
                            if bit_config else args.weight_bits),
            "activation_bits": (sorted({a for _, a in bit_config.values()})
                                if bit_config else args.activation_bits),
            "quantization": "symmetric uniform, zero-point=0",
            "weight_granularity": "per-output-channel",
            "activation_granularity": "per-tensor (per-layer input)",
            "fp32_kept": "BatchNorm params/stats, biases, final logits",
            "calibration_images": n_calib,
            "num_quantized_layers": len(wrappers),
        },
        "accuracy": {
            "fp32_top1": fp32_acc,
            "quantized_top1": q_acc,
            "quantized_loss": q_loss,
            "top1_drop": (fp32_acc - q_acc) if fp32_acc is not None else None,
            "per_class_top1": per_class,
        },
        "weight_memory": w_rep,
        "activation_memory": a_rep,
    }
    if args.per_layer:
        results["per_layer"] = per_layer_report(wrappers, w_spec, a_spec)

    tag = args.tag or (args.mixed_config or
                       f"w{args.weight_bits}a{args.activation_bits}")
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- summary -------------------------------------------")
    print(f"1. FP32 model size          : {w_rep['fp32_model_size_mb']:.3f} MB")
    print(f"2. compressed weight size   : {w_rep['compressed_weight_mb']:.3f} MB"
          f"  (+{w_rep['fp32_kept_bytes']/2**10:.1f} KB FP32 BN/bias)")
    print(f"3. activation memory        : {a_rep['activation_quant_mb']:.4f} MB"
          f"  (FP32 {a_rep['activation_fp32_mb']:.4f} MB, "
          f"batch {a_rep['batch_size']})")
    print(f"4. scale/metadata overhead  : "
          f"{w_rep['scale_metadata_bytes']/2**10:.2f} KB")
    print(f"5. weight compression ratio : "
          f"{w_rep['weight_compression_ratio']:.2f}x (whole model), "
          f"{w_rep['weight_only_compression_ratio']:.2f}x (quantized weights)")
    print(f"6. activation compression   : "
          f"{a_rep['activation_compression_ratio']:.2f}x")
    print(f"7. test accuracy            : {q_acc*100:.2f}%")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
