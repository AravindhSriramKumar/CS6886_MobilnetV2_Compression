"""
Prediction-level comparison across precision settings (Q3 analysis / Q4 material).

Runs the same CIFAR-10 test images through several configs -- FP32 baseline,
uniform quantization, and the HAWQ-V2 mixed-precision config -- keeping every
logit, then reports:

  * per-config accuracy and flips vs FP32 (correct->wrong and wrong->correct)
  * the pairwise disagreement structure between two quantized configs, split
    into shared errors vs errors unique to each -- this is what shows whether
    a bit allocation actually fixes errors rather than relocating them
  * confidence/margin degradation on the images that do *not* flip
  * a per-image drill-down (logits + softmax side by side)

A single image is an illustration, never evidence: pick it from the flip lists
this script prints, not at random.

    python src/predict_compare.py --configs uniform_w8a8 uniform_w4a8 hawq_w8-4_top25_a8
    python src/predict_compare.py --image-index 1234
    python src/predict_compare.py --show-flips 5 --reference uniform_w4a8 \
        --candidate hawq_w8-4_top25_a8
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from compress import load_baseline
from data import CLASSES, get_dataloaders
from quantize import apply_bit_config, calibrate, set_mode, wrap_model


@torch.no_grad()
def collect_logits(model, loader, device):
    """All test logits + labels, in dataset order (loader must not shuffle)."""
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def load_bit_assignments(path):
    with open(path) as f:
        return {k: {n: tuple(v) for n, v in cfg.items()}
                for k, cfg in json.load(f)["bit_assignments"].items()}


def flip_stats(ref_logits, cand_logits, labels):
    """How `cand` differs from `ref` at the prediction level."""
    ref_pred = ref_logits.argmax(1)
    cand_pred = cand_logits.argmax(1)
    ref_ok = ref_pred == labels
    cand_ok = cand_pred == labels
    changed = ref_pred != cand_pred

    return {
        "ref_accuracy": ref_ok.float().mean().item(),
        "cand_accuracy": cand_ok.float().mean().item(),
        "predictions_changed": int(changed.sum()),
        "correct_to_wrong": int((ref_ok & ~cand_ok).sum()),
        "wrong_to_correct": int((~ref_ok & cand_ok).sum()),
        "wrong_to_wrong_changed": int((~ref_ok & ~cand_ok & changed).sum()),
        "both_correct": int((ref_ok & cand_ok).sum()),
        "both_wrong": int((~ref_ok & ~cand_ok).sum()),
        "agreement": (ref_pred == cand_pred).float().mean().item(),
    }


def margin(logits):
    """Top-1 minus top-2 logit: how close the decision was to flipping."""
    top2 = logits.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def confidence_stats(fp32_logits, q_logits, labels):
    """Damage on the images that did NOT flip -- the pre-flip warning signal."""
    same = fp32_logits.argmax(1) == q_logits.argmax(1)
    fp32_p = F.softmax(fp32_logits, 1)
    q_p = F.softmax(q_logits, 1)
    idx = torch.arange(len(labels))
    return {
        "unflipped_images": int(same.sum()),
        "mean_true_class_prob_fp32": fp32_p[idx, labels][same].mean().item(),
        "mean_true_class_prob_quant": q_p[idx, labels][same].mean().item(),
        "mean_margin_fp32": margin(fp32_logits)[same].mean().item(),
        "mean_margin_quant": margin(q_logits)[same].mean().item(),
        "mean_logit_l2_shift": (fp32_logits - q_logits).norm(dim=1).mean().item(),
    }


def show_image(index, all_logits, labels, order):
    print(f"\n=== test image #{index} -- true label: {CLASSES[labels[index]]} ===")
    head = f"{'config':<22}{'pred':<12}{'p(pred)':>9}{'p(true)':>9}{'margin':>9}"
    print(head)
    print("-" * len(head))
    for name in order:
        lg = all_logits[name][index]
        p = F.softmax(lg, 0)
        pred = int(lg.argmax())
        ok = "OK " if pred == labels[index] else "ERR"
        print(f"{name:<22}{ok} {CLASSES[pred]:<8}{p[pred]:>9.3f}"
              f"{p[labels[index]]:>9.3f}{margin(lg[None])[0]:>9.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="runs/baseline/best.pt")
    p.add_argument("--assignments", type=str,
                   default="runs/mixed_precision/results.json")
    p.add_argument("--configs", type=str, nargs="+",
                   default=["uniform_w8a8", "uniform_w4a8",
                            "hawq_w8-4_top25_a8"])
    p.add_argument("--reference", type=str, default="uniform_w4a8",
                   help="config compared against --candidate pairwise")
    p.add_argument("--candidate", type=str, default="hawq_w8-4_top25_a8")
    p.add_argument("--image-index", type=int, nargs="*", default=None,
                   help="test-set indices to print a per-image breakdown for")
    p.add_argument("--show-flips", type=int, default=5,
                   help="how many example images to print from each flip class")
    p.add_argument("--calib-batches", type=int, default=8)
    p.add_argument("--calib-batch-size", type=int, default=64)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--output", type=str,
                   default="runs/mixed_precision/prediction_comparison.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    assignments = load_bit_assignments(args.assignments)
    for name in set(args.configs + [args.reference, args.candidate]):
        if name not in assignments:
            raise SystemExit(f"unknown config '{name}'. available: "
                             f"{', '.join(assignments)}")

    model, _ = load_baseline(args.checkpoint, device)
    _, test_loader = get_dataloaders(root=args.data_root,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     augment=False, seed=args.seed)

    wrappers = wrap_model(model, 8, 8)
    model.to(device)
    set_mode(wrappers, "fp32")
    # FP32 pass first: the wrapper is a passthrough in 'fp32' mode, so this is
    # exactly the Q1 baseline, evaluated on identical batches.
    calib_loader, _ = get_dataloaders(root=args.data_root,
                                      batch_size=args.calib_batch_size,
                                      num_workers=args.num_workers,
                                      augment=False, seed=args.seed)
    calibrate(model, wrappers, calib_loader, device, args.calib_batches)

    all_logits = {}
    set_mode(wrappers, "fp32")
    all_logits["fp32_baseline"], labels = collect_logits(model, test_loader,
                                                        device)
    for name in args.configs:
        apply_bit_config(wrappers, assignments[name])
        set_mode(wrappers, "quant")
        all_logits[name], _ = collect_logits(model, test_loader, device)

    order = ["fp32_baseline"] + list(args.configs)
    fp32 = all_logits["fp32_baseline"]

    print(f"\ntest images: {len(labels)}")
    print("\n--- vs FP32 baseline ------------------------------------------")
    hdr = (f"{'config':<22}{'acc':>8}{'changed':>9}{'C->W':>7}{'W->C':>7}"
           f"{'agree':>8}{'p(true)':>9}{'margin':>8}")
    print(hdr)
    print("-" * len(hdr))
    print(f"{'fp32_baseline':<22}"
          f"{(fp32.argmax(1) == labels).float().mean()*100:>7.2f}%"
          f"{'-':>9}{'-':>7}{'-':>7}{'-':>8}"
          f"{F.softmax(fp32,1)[torch.arange(len(labels)), labels].mean():>9.3f}"
          f"{margin(fp32).mean():>8.3f}")

    report = {"num_images": len(labels), "vs_fp32": {}, "pairwise": {}}
    for name in args.configs:
        fs = flip_stats(fp32, all_logits[name], labels)
        cs = confidence_stats(fp32, all_logits[name], labels)
        report["vs_fp32"][name] = {**fs, **cs}
        print(f"{name:<22}{fs['cand_accuracy']*100:>7.2f}%"
              f"{fs['predictions_changed']:>9}{fs['correct_to_wrong']:>7}"
              f"{fs['wrong_to_correct']:>7}{fs['agreement']*100:>7.1f}%"
              f"{cs['mean_true_class_prob_quant']:>9.3f}"
              f"{cs['mean_margin_quant']:>8.3f}")

    # --- pairwise: does the mixed config fix errors or just move them? ---
    ref, cand = args.reference, args.candidate
    r_pred = all_logits[ref].argmax(1)
    c_pred = all_logits[cand].argmax(1)
    r_ok, c_ok = r_pred == labels, c_pred == labels
    only_ref_wrong = (~r_ok & c_ok)
    only_cand_wrong = (r_ok & ~c_ok)
    both_wrong = (~r_ok & ~c_ok)
    report["pairwise"] = {
        "reference": ref, "candidate": cand,
        "reference_errors": int((~r_ok).sum()),
        "candidate_errors": int((~c_ok).sum()),
        "shared_errors": int(both_wrong.sum()),
        "fixed_by_candidate": int(only_ref_wrong.sum()),
        "broken_by_candidate": int(only_cand_wrong.sum()),
        "net_gain": int(only_ref_wrong.sum() - only_cand_wrong.sum()),
        "agreement": (r_pred == c_pred).float().mean().item(),
    }
    pw = report["pairwise"]
    print(f"\n--- {cand}  vs  {ref} ---")
    print(f"  errors: {pw['reference_errors']} ({ref}) vs "
          f"{pw['candidate_errors']} ({cand}); {pw['shared_errors']} shared")
    print(f"  fixed by {cand}: {pw['fixed_by_candidate']}   "
          f"broken: {pw['broken_by_candidate']}   "
          f"net: {pw['net_gain']:+d} images "
          f"({pw['net_gain']/len(labels)*100:+.2f} pt)")
    print(f"  prediction agreement: {pw['agreement']*100:.2f}%")

    # --- illustrative images, drawn from the flip sets --------------------
    fixed_idx = only_ref_wrong.nonzero().flatten().tolist()
    broken_idx = only_cand_wrong.nonzero().flatten().tolist()
    report["example_indices"] = {
        "fixed_by_candidate": fixed_idx[:50],
        "broken_by_candidate": broken_idx[:50],
    }
    if args.show_flips:
        print(f"\nexample images {cand} gets RIGHT and {ref} gets WRONG "
              f"(indices: {fixed_idx[:args.show_flips]})")
        for i in fixed_idx[:args.show_flips]:
            show_image(i, all_logits, labels, order)
        if broken_idx:
            print(f"\nexample images {cand} gets WRONG and {ref} gets RIGHT "
                  f"(indices: {broken_idx[:args.show_flips]})")
            for i in broken_idx[:args.show_flips]:
                show_image(i, all_logits, labels, order)

    for i in (args.image_index or []):
        show_image(i, all_logits, labels, order)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {args.output}")


if __name__ == "__main__":
    main()
