"""
Q3: HAWQ-V2-inspired sensitivity-guided mixed-precision quantization.

Pipeline (all on top of the untouched Q1 checkpoint and the manual Q2
quantizer -- no quantization/compression library anywhere):

  1. Hutchinson Hessian-trace estimation per Conv/Linear layer  (sensitivity.py)
  2. HAWQ-V2 score  Omega_l = |Tr(H_l)/n_l| * ||W_l - Q_b(W_l)||^2, ranked
  3. top-`frac` most sensitive layers -> 8 bit, the rest -> 4 or 2 bit
  4. one calibration pass, then every config is evaluated by re-targeting the
     already-wrapped model with `quantize.apply_bit_config`
  5. accuracy + weight/activation compression + compressed size per config
  6. Pareto front over (accuracy, total compression)
  7. Weights & Biases logging + a Parallel Coordinates plot
  8. CSV + JSON dumps

    python src/mixed_precision.py --checkpoint runs/baseline/best.pt \
        --wandb-mode offline
"""

import argparse
import csv
import json
import os
import time

import torch

from compress import load_baseline
from data import get_dataloaders
from evaluate import evaluate
from memory import activation_report, state_dict_elements, weight_report
from quantize import apply_bit_config, calibrate, set_mode, wrap_model
from sensitivity import hessian_traces, rank_layers, sensitivity_scores


# --------------------------------------------------------------------------
# Mixed-precision assignment
# --------------------------------------------------------------------------
def make_bit_config(ranking, frac_high, low_bits, high_bits=8,
                    act_policy="mixed", fixed_act_bits=8):
    """Assign bits: top `frac_high` of the sensitivity ranking -> `high_bits`.

    act_policy:
      'mixed'   -- activations follow the same sensitivity split as weights
      'fixed'   -- every layer's activations stay at `fixed_act_bits`
    """
    n_high = max(1, round(frac_high * len(ranking)))
    high = set(ranking[:n_high])
    cfg = {}
    for name in ranking:
        wb = high_bits if name in high else low_bits
        if act_policy == "fixed":
            ab = fixed_act_bits
        else:
            ab = high_bits if name in high else low_bits
        cfg[name] = (wb, ab)
    return cfg, n_high


def build_configs(ranking, args):
    """The sweep: uniform reference points + sensitivity-guided mixed ones."""
    configs = []
    for wb, ab in [(32, 32), (8, 8), (4, 8), (4, 4), (2, 8)]:
        configs.append({
            "name": f"uniform_w{wb}a{ab}",
            "policy": "uniform",
            "low_bits": wb, "high_bits": wb, "frac_high": 1.0,
            "act_policy": "uniform",
            "bit_config": {n: (wb, ab) for n in ranking},
            "n_high": len(ranking),
        })
    for low in args.low_bits:
        for frac in args.frac_high:
            for act_policy in args.act_policies:
                cfg, n_high = make_bit_config(
                    ranking, frac, low, args.high_bits, act_policy,
                    args.fixed_act_bits)
                tag = ("a8" if act_policy == "fixed" else "amix")
                configs.append({
                    "name": f"hawq_w{args.high_bits}-{low}_top{int(frac*100)}_{tag}",
                    "policy": "hawq-v2",
                    "low_bits": low, "high_bits": args.high_bits,
                    "frac_high": frac, "act_policy": act_policy,
                    "bit_config": cfg, "n_high": n_high,
                })
    return configs


# --------------------------------------------------------------------------
# Evaluation of one config
# --------------------------------------------------------------------------
def eval_config(cfg, model, wrappers, baseline_numel, test_loader, device,
                mem_batch_size):
    apply_bit_config(wrappers, cfg["bit_config"])
    set_mode(wrappers, "quant")
    t0 = time.time()
    loss, acc, _ = evaluate(model, test_loader, device)

    wb_spec = {n: cfg["bit_config"][n] for n in cfg["bit_config"]}
    w_rep = weight_report(wrappers, baseline_numel, wb_spec)
    a_rep = activation_report(wrappers, wb_spec, batch_size=mem_batch_size)

    total_fp32 = w_rep["fp32_model_size_bytes"] + a_rep["activation_fp32_bytes"]
    total_comp = w_rep["compressed_total_bytes"] + a_rep["activation_quant_bytes"]

    bits = [b for b, _ in cfg["bit_config"].values()]
    abits = [a for _, a in cfg["bit_config"].values()]
    return {
        "config": cfg["name"],
        "policy": cfg["policy"],
        "high_bits": cfg["high_bits"],
        "low_bits": cfg["low_bits"],
        "frac_high": cfg["frac_high"],
        "act_policy": cfg["act_policy"],
        "num_high_precision_layers": cfg["n_high"],
        "avg_weight_bits": sum(bits) / len(bits),
        "avg_activation_bits": sum(abits) / len(abits),
        "test_accuracy": acc,
        "test_loss": loss,
        "compressed_model_mb": w_rep["compressed_total_mb"],
        "compressed_weight_mb": w_rep["compressed_weight_mb"],
        "fp32_model_mb": w_rep["fp32_model_size_mb"],
        "scale_metadata_kb": w_rep["scale_metadata_bytes"] / 2**10,
        "weight_compression_ratio": w_rep["weight_compression_ratio"],
        "weight_only_compression_ratio": w_rep["weight_only_compression_ratio"],
        "activation_mb": a_rep["activation_quant_mb"],
        "activation_compression_ratio": a_rep["activation_compression_ratio"],
        "total_compression_ratio": total_fp32 / total_comp,
        "eval_seconds": time.time() - t0,
    }


# --------------------------------------------------------------------------
# Pareto front
# --------------------------------------------------------------------------
def pareto_front(rows, x="total_compression_ratio", y="test_accuracy"):
    """Rows not dominated on (compression, accuracy) -- both maximized."""
    front = []
    for r in rows:
        dominated = any(o is not r and o[x] >= r[x] and o[y] >= r[y]
                        and (o[x] > r[x] or o[y] > r[y]) for o in rows)
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: -r[y])


# --------------------------------------------------------------------------
# W&B
# --------------------------------------------------------------------------
PARALLEL_DIMS = ["avg_weight_bits", "avg_activation_bits", "frac_high",
                 "weight_compression_ratio", "activation_compression_ratio",
                 "compressed_model_mb", "total_compression_ratio",
                 "test_accuracy"]


def parallel_coordinates_chart(table, columns, title):
    """W&B's built-in parallel-coordinates Vega preset.

    `wandb.plot.parallel_coordinates` was removed from the public API in
    recent wandb versions, so the preset is requested directly through
    `plot_table` (same chart, same spec id).
    """
    import wandb
    return wandb.plot.plot_table(
        vega_spec_name="wandb/parallel_coordinates/v0",
        data_table=table,
        fields={"columns": columns},
        string_fields={"title": title},
    )


def log_wandb(rows, front, args, ranking, scores):
    """One W&B run per config + a summary run carrying the parallel-coords plot."""
    import wandb

    os.environ.setdefault("WANDB_SILENT", "true")
    front_names = {r["config"] for r in front}

    for r in rows:
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         group="q3-mixed-precision", name=r["config"],
                         mode=args.wandb_mode, reinit=True,
                         config={k: r[k] for k in
                                 ("policy", "high_bits", "low_bits",
                                  "frac_high", "act_policy")})
        wandb.log({**{k: v for k, v in r.items() if isinstance(v, (int, float))},
                   "on_pareto_front": int(r["config"] in front_names)})
        run.finish()

    summary = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         group="q3-mixed-precision", name="q3-summary",
                         mode=args.wandb_mode, reinit=True)
    table = wandb.Table(columns=["config"] + PARALLEL_DIMS + ["on_pareto_front"])
    for r in rows:
        table.add_data(r["config"], *[r[d] for d in PARALLEL_DIMS],
                       int(r["config"] in front_names))
    wandb.log({
        "sweep_table": table,
        "parallel_coordinates": parallel_coordinates_chart(
            table, PARALLEL_DIMS,
            "Q3 mixed precision: bits vs compression vs accuracy"),
        "accuracy_vs_compression": wandb.plot.scatter(
            table, "total_compression_ratio", "test_accuracy",
            title="Accuracy vs total compression"),
        "layer_sensitivity": wandb.plot.bar(
            wandb.Table(data=[[n, scores[n]] for n in ranking],
                        columns=["layer", "hawq_score"]),
            "layer", "hawq_score", title="HAWQ-V2 layer sensitivity (ranked)"),
    })
    summary.finish()
    return summary.url if args.wandb_mode == "online" else args.wandb_mode


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="runs/baseline/best.pt")
    p.add_argument("--high-bits", type=int, default=8)
    p.add_argument("--low-bits", type=int, nargs="+", default=[4, 2])
    p.add_argument("--frac-high", type=float, nargs="+",
                   default=[0.1, 0.25, 0.5, 0.75])
    p.add_argument("--act-policies", type=str, nargs="+",
                   default=["fixed", "mixed"])
    p.add_argument("--fixed-act-bits", type=int, default=8)
    p.add_argument("--hessian-batches", type=int, default=8)
    p.add_argument("--hessian-samples", type=int, default=8)
    p.add_argument("--hessian-batch-size", type=int, default=64)
    p.add_argument("--rank-bits", type=int, default=4,
                   help="bit-width used for the perturbation term of the score")
    p.add_argument("--calib-batches", type=int, default=8)
    p.add_argument("--calib-batch-size", type=int, default=64)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--mem-batch-size", type=int, default=1)
    p.add_argument("--output-dir", type=str, default="runs/mixed_precision")
    p.add_argument("--wandb-mode", type=str, default="offline",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-project", type=str, default="cs6886-a2-compression")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    model, ckpt = load_baseline(args.checkpoint, device)
    calib_loader, test_loader = get_dataloaders(
        root=args.data_root, batch_size=args.hessian_batch_size,
        num_workers=args.num_workers, augment=False, seed=args.seed)
    _, test_loader = get_dataloaders(
        root=args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers, augment=False, seed=args.seed)

    # --- 1/2. sensitivity ------------------------------------------------
    print(f"estimating Hessian traces "
          f"({args.hessian_batches} batches x {args.hessian_samples} Hutchinson draws)")
    traces, avg_traces = hessian_traces(
        model, calib_loader, device, args.hessian_batches,
        args.hessian_samples, seed=args.seed)
    scores = sensitivity_scores(model, avg_traces, args.rank_bits)
    ranking = rank_layers(scores)
    print("top-8 most sensitive layers:")
    for n in ranking[:8]:
        print(f"  {n:38s} avg_trace={avg_traces[n]:+.4e}  omega={scores[n]:.4e}")

    with open(os.path.join(args.output_dir, "sensitivity.json"), "w") as f:
        json.dump({"rank_bits": args.rank_bits,
                   "hessian_batches": args.hessian_batches,
                   "hessian_samples": args.hessian_samples,
                   "ranking": ranking,
                   "hessian_trace": traces,
                   "avg_hessian_trace": avg_traces,
                   "hawq_score": scores}, f, indent=2)

    # --- 3/4. wrap + calibrate once --------------------------------------
    baseline_numel = state_dict_elements(model)
    wrappers = wrap_model(model, 8, 8)
    model.to(device)
    calib_loader, _ = get_dataloaders(root=args.data_root,
                                      batch_size=args.calib_batch_size,
                                      num_workers=args.num_workers,
                                      augment=False, seed=args.seed)
    n_calib = calibrate(model, wrappers, calib_loader, device,
                        args.calib_batches)
    print(f"calibrated activation ranges on {n_calib} images "
          f"({len(wrappers)} layers)")

    # --- 5. sweep --------------------------------------------------------
    configs = build_configs(ranking, args)
    rows = []
    print(f"\nevaluating {len(configs)} configurations")
    for i, cfg in enumerate(configs, 1):
        r = eval_config(cfg, model, wrappers, baseline_numel, test_loader,
                        device, args.mem_batch_size)
        rows.append(r)
        print(f"[{i:2d}/{len(configs)}] {r['config']:34s} "
              f"acc {r['test_accuracy']*100:5.2f}%  "
              f"W {r['weight_compression_ratio']:5.2f}x  "
              f"A {r['activation_compression_ratio']:5.2f}x  "
              f"size {r['compressed_model_mb']:.3f} MB")

    # --- 6. Pareto -------------------------------------------------------
    front = pareto_front(rows)
    print("\nPareto-optimal configurations (accuracy vs total compression):")
    for r in front:
        print(f"  {r['config']:34s} acc {r['test_accuracy']*100:5.2f}%  "
              f"total {r['total_compression_ratio']:5.2f}x  "
              f"model {r['compressed_model_mb']:.3f} MB")

    # Best trade-off for Q4: highest compression within 1 pt of FP32 accuracy.
    fp32_acc = next(r["test_accuracy"] for r in rows
                    if r["config"] == "uniform_w32a32")
    eligible = [r for r in front if fp32_acc - r["test_accuracy"] <= 0.01]
    best = max(eligible or front, key=lambda r: r["total_compression_ratio"])
    print(f"\nrecommended for Q4: {best['config']}  "
          f"acc {best['test_accuracy']*100:.2f}% "
          f"(-{(fp32_acc-best['test_accuracy'])*100:.2f} pt), "
          f"{best['total_compression_ratio']:.2f}x total, "
          f"{best['compressed_model_mb']:.3f} MB")

    # --- 7/8. logging + dumps --------------------------------------------
    front_names = {r["config"] for r in front}
    payload = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "fp32_accuracy": fp32_acc,
        "calibration_images": n_calib,
        "sensitivity_ranking": ranking,
        "results": rows,
        "pareto_front": [r["config"] for r in front],
        "recommended_for_q4": best["config"],
        "bit_assignments": {c["name"]: {k: list(v) for k, v in
                                        c["bit_config"].items()}
                            for c in configs},
    }
    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    csv_path = os.path.join(args.output_dir, "results.csv")
    cols = list(rows[0].keys()) + ["on_pareto_front"]
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=cols)
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow({**r,
                           "on_pareto_front": int(r["config"] in front_names)})

    if args.wandb_mode != "disabled":
        where = log_wandb(rows, front, args, ranking, scores)
        print(f"logged {len(rows)} runs + parallel-coordinates plot to W&B ({where})")
    print(f"saved -> {json_path}\nsaved -> {csv_path}")


if __name__ == "__main__":
    main()
