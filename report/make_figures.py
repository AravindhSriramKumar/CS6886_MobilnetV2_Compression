"""
Regenerates every figure used by report/main.tex from artefacts already present
in the repository. Run from the repository root:

    python report/make_figures.py

Sources: runs/baseline/history.csv, runs/baseline/best.pt,
runs/compression/*.json, runs/mixed_precision/sensitivity.json,
runs/augmented_data_q1/*.png, docs/wandb.png.
"""
import csv, glob, json, os, shutil, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, "src")
OUT = "report/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5,
    "axes.titlesize": 9.5, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C = {"train": "#1f4e79", "test": "#c1121f"}

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)

# --- Fig 1: augmentation montage (reuses runs/augmented_data_q1/*.png) -------
files = [sorted(glob.glob("runs/augmented_data_q1/*.png"))[i] for i in (0, 1, 2)][:2]
fig, axes = plt.subplots(len(files), 1, figsize=(5.6, 1.20 * len(files)))
for ax, f in zip(np.atleast_1d(axes), files):
    ax.imshow(mpimg.imread(f)); ax.axis("off")
fig.subplots_adjust(hspace=0.02)
save(fig, "augmentation_examples")

# --- Fig 2/3: training curves from history.csv ------------------------------
h = {k: [] for k in ["epoch", "train_loss", "train_acc", "test_loss", "test_acc"]}
with open("runs/baseline/history.csv") as f:
    for r in csv.DictReader(f):
        for k in h:
            h[k].append(float(r[k]))
ep = np.array(h["epoch"])

fig, ax = plt.subplots(figsize=(4.3, 2.4))
ax.plot(ep, h["train_loss"], color=C["train"], lw=1.2, label="Train loss")
ax.plot(ep, h["test_loss"], color=C["test"], lw=1.2, label="Test loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss (label smoothing 0.1)")
ax.set_xlim(1, ep[-1]); ax.legend(frameon=False)
save(fig, "loss_curve")

best_i = int(np.argmax(h["test_acc"]))
fig, ax = plt.subplots(figsize=(4.3, 2.4))
ax.plot(ep, 100 * np.array(h["train_acc"]), color=C["train"], lw=1.2, label="Train top-1")
ax.plot(ep, 100 * np.array(h["test_acc"]), color=C["test"], lw=1.2, label="Test top-1")
ax.scatter([ep[best_i]], [100 * h["test_acc"][best_i]], s=22, color="k", zorder=5)
ax.annotate(f"best: {100*h['test_acc'][best_i]:.2f}% @ epoch {int(ep[best_i])}",
            (ep[best_i], 100 * h["test_acc"][best_i]), textcoords="offset points",
            xytext=(-8, -24), ha="right", fontsize=7.5)
ax.set_xlabel("Epoch"); ax.set_ylabel("Top-1 accuracy (%)")
ax.set_xlim(1, ep[-1]); ax.set_ylim(20, 100); ax.legend(frameon=False, loc="lower right")
save(fig, "accuracy_curve")

# --- Fig 4: confusion matrix (no per-cell numbers) --------------------------
from data import CLASSES
CM = "report/figures/confusion_counts.npy"
if not os.path.exists(CM):
    import torch
    from model import build_model
    from data import get_dataloaders
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load("runs/baseline/best.pt", map_location=dev, weights_only=False)
    ta = ck.get("args", {})
    m = build_model(num_classes=10, width_mult=ta.get("width_mult", 1.0),
                    dropout=ta.get("dropout", 0.2)).to(dev)
    m.load_state_dict(ck["model_state"]); m.eval()
    _, tl = get_dataloaders(root="./data", batch_size=256, num_workers=4, augment=False)
    cm = np.zeros((10, 10), dtype=np.int64)
    with torch.no_grad():
        for x, y in tl:
            p = m(x.to(dev)).argmax(1).cpu().numpy()
            for t, pr in zip(y.numpy(), p):
                cm[t, pr] += 1
    np.save(CM, cm)
    print("checkpoint accuracy:", np.trace(cm) / cm.sum())
cm = np.load(CM)
cmn = cm / cm.sum(1, keepdims=True)

fig, ax = plt.subplots(figsize=(4.6, 4.0))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels(CLASSES, rotation=45, ha="left", fontsize=7)
ax.set_yticklabels(CLASSES, fontsize=7)
ax.set_xlabel("Prediction", fontsize=9, labelpad=8)
ax.set_ylabel("Actual", fontsize=9, labelpad=6)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(True); spine.set_linewidth(0.6); spine.set_color("0.3")
thresh = cm.max() / 2.0
for i in range(10):
    for j in range(10):
        ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center", fontsize=5.6,
                color="white" if cm[i, j] > thresh else "#1f4e79")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("Test images", fontsize=8); cb.ax.tick_params(labelsize=7)
save(fig, "confusion_matrix")
off = sorted(((100 * cmn[i, j], CLASSES[i], CLASSES[j])
              for i in range(10) for j in range(10) if i != j), reverse=True)
print("top confusions:", [(round(v, 1), a, b) for v, a, b in off[:4]])

# --- Fig 5: uniform sweep, two panels ---------------------------------------
rows = []
for f in sorted(glob.glob("runs/compression/w*.json")):
    d = json.load(open(f))
    rows.append(dict(w=d["config"]["weight_bits"], a=d["config"]["activation_bits"],
                     acc=100 * d["accuracy"]["quantized_top1"],
                     mr=d["weight_memory"]["weight_compression_ratio"]))
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.6))
ax = axs[0]
hi = sorted([r for r in rows if r["a"] >= 8], key=lambda r: r["w"])
lo = sorted([r for r in rows if r["a"] < 8], key=lambda r: r["w"])
ax.plot([r["w"] for r in hi], [r["acc"] for r in hi], "o-", color=C["train"],
        lw=1.4, ms=5, label="8-/32-bit activations")
ax.plot([r["w"] for r in lo], [r["acc"] for r in lo], "s--", color=C["test"],
        lw=1.4, ms=5, label="4-/2-bit activations")
for r in rows:
    off = (-4, 5) if r["w"] == 32 else (5, 4)
    ax.annotate(f"W{r['w']}A{r['a']}", (r["w"], r["acc"]), ha="right" if r["w"] == 32 else "left",
                textcoords="offset points", xytext=off, fontsize=6.5)
ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 32])
ax.set_xticklabels(["2", "4", "8", "32"])
ax.axhline(93.18, color="gray", ls=":", lw=0.9)
ax.text(2.1, 82, "FP32 baseline 93.18%", fontsize=7, color="gray")
ax.set_xlabel("Weight bit-width $b_W$"); ax.set_ylabel("Test top-1 accuracy (%)")
ax.set_ylim(0, 102); ax.legend(frameon=False, loc="center right", fontsize=7)

ax = axs[1]
for r in rows:
    ax.scatter(r["mr"], r["acc"], marker="o" if r["a"] >= 8 else "s", s=36,
               color=C["train"] if r["a"] >= 8 else C["test"], zorder=4)
    off = (-5, 6 if r["a"] >= 8 else -10) if r["mr"] > 9 else (5, 4)
    ax.annotate(f"W{r['w']}A{r['a']}", (r["mr"], r["acc"]), ha="right" if r["mr"] > 9 else "left",
                textcoords="offset points", xytext=off, fontsize=6.5)
ax.set_xlabel(r"Whole-model compression ratio ($\times$)")
ax.set_ylabel("Test top-1 accuracy (%)"); ax.set_ylim(0, 102); ax.set_xlim(0, 12.5)
save(fig, "q2_uniform_sweep")

# --- Fig 6: HAWQ-V2 layer sensitivity ---------------------------------------
s = json.load(open("runs/mixed_precision/sensitivity.json"))
top = s["ranking"][:20]
vals = [s["hawq_score"][l] for l in top]
short = [l.replace("features.", "f").replace("classifier.", "cls.").replace(".conv", "")
         for l in top]
fig, ax = plt.subplots(figsize=(4.6, 2.9))
ax.barh(range(len(top)), vals,
        color=["#c1121f" if i < 13 else "#9fb3c8" for i in range(len(top))], height=0.72)
ax.set_yticks(range(len(top))); ax.set_yticklabels(short, fontsize=6.6)
ax.invert_yaxis(); ax.set_xscale("log")
ax.set_xlabel(r"HAWQ-V2 sensitivity $\Omega_l = |\mathrm{Tr}(H_l)/n_l| \cdot "
              r"\|W_l - Q_4(W_l)\|_2^2$")
ax.axhline(12.5, color="k", ls="--", lw=0.9)
ax.text(0.985, 0.045, "below: 4-bit weights", transform=ax.transAxes,
        ha="right", fontsize=7)
ax.set_title("top 13 of 53 layers (top-25%) $\\rightarrow$ 8-bit weights; "
             "remainder $\\rightarrow$ 4-bit", fontsize=7.5, color="0.2", pad=4)
save(fig, "hawq_sensitivity")

# --- Fig 7: W&B parallel coordinates (copied from docs/) --------------------
shutil.copyfile("docs/wandb.png", f"{OUT}/wandb_parallel_coordinates.png")
print("copied docs/wandb.png -> wandb_parallel_coordinates.png")
