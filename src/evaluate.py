"""
Standalone evaluation for CS6886 Assignment 2 -- loads a checkpoint and
reports top-1 test accuracy. Kept separate from train.py (Q5 modularity:
training / evaluation / compression are distinct stages), and reused later
to score compressed checkpoints (Q4).

    python src/evaluate.py --checkpoint runs/baseline/best.pt
"""

import argparse

import torch
import torch.nn as nn

from data import get_dataloaders, CLASSES
from model import build_model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_n = 0.0, 0, 0
    per_class_correct = torch.zeros(len(CLASSES))
    per_class_total = torch.zeros(len(CLASSES))

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        preds = logits.argmax(1)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (preds == y).sum().item()
        total_n += bs

        for c in range(len(CLASSES)):
            mask = y == c
            per_class_total[c] += mask.sum().item()
            per_class_correct[c] += (preds[mask] == c).sum().item()

    per_class_acc = (per_class_correct / per_class_total.clamp(min=1)).tolist()
    return total_loss / total_n, total_correct / total_n, per_class_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})

    model = build_model(num_classes=10,
                        width_mult=train_args.get("width_mult", 1.0),
                        dropout=train_args.get("dropout", 0.2)).to(device)
    model.load_state_dict(ckpt["model_state"])

    _, test_loader = get_dataloaders(root=args.data_root,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     augment=False)

    loss, acc, per_class_acc = evaluate(model, test_loader, device)
    print(f"checkpoint: {args.checkpoint} (trained epoch {ckpt.get('epoch')})")
    print(f"test loss: {loss:.4f}")
    print(f"test top-1 accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("per-class accuracy:")
    for cls, a in zip(CLASSES, per_class_acc):
        print(f"  {cls:>10s}: {a*100:.2f}%")


if __name__ == "__main__":
    main()
