"""
Plot loss/accuracy curves from a train.py history.csv, for the Q1(c) write-up.

    python src/plot_curves.py --history runs/baseline/history.csv
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_history(path):
    rows = {"epoch": [], "train_loss": [], "train_acc": [],
            "test_loss": [], "test_acc": []}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k in rows:
                rows[k].append(float(row[k]))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=str, default="./runs/baseline/history.csv")
    p.add_argument("--output-dir", type=str, default=None,
                   help="defaults to the history file's directory")
    args = p.parse_args()

    out_dir = args.output_dir or os.path.dirname(args.history)
    os.makedirs(out_dir, exist_ok=True)
    h = load_history(args.history)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(h["epoch"], h["train_loss"], label="train loss")
    ax.plot(h["epoch"], h["test_loss"], label="test loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("MobileNet-v2 / CIFAR-10 -- loss curves")
    ax.legend()
    fig.tight_layout()
    loss_path = os.path.join(out_dir, "loss_curve.png")
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(h["epoch"], [a * 100 for a in h["train_acc"]], label="train acc")
    ax.plot(h["epoch"], [a * 100 for a in h["test_acc"]], label="test acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("top-1 accuracy (%)")
    ax.set_title("MobileNet-v2 / CIFAR-10 -- accuracy curves")
    ax.legend()
    fig.tight_layout()
    acc_path = os.path.join(out_dir, "accuracy_curve.png")
    fig.savefig(acc_path, dpi=150)
    plt.close(fig)

    best_epoch = max(range(len(h["test_acc"])), key=lambda i: h["test_acc"][i])
    print(f"wrote {loss_path}")
    print(f"wrote {acc_path}")
    print(f"best test acc: {h['test_acc'][best_epoch]*100:.2f}% "
          f"at epoch {int(h['epoch'][best_epoch])}")
    print(f"final train acc: {h['train_acc'][-1]*100:.2f}%, "
          f"final test acc: {h['test_acc'][-1]*100:.2f}%")


if __name__ == "__main__":
    main()
