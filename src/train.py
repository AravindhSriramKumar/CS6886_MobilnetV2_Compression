"""
Training script for CS6886 Assignment 2 (Q1b/Q1c) -- baseline MobileNet-v2
on CIFAR-10, no compression.

    python src/train.py --epochs 180

Writes, under --output-dir:
    history.csv   -- per-epoch train/test loss & accuracy
    best.pt       -- checkpoint with the highest test accuracy seen
    last.pt       -- checkpoint after the final epoch
"""

import argparse
import csv
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from data import get_dataloaders
from model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_lr_scheduler(optimizer, epochs, warmup_epochs, steps_per_epoch, base_lr):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, total_correct, total_n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train_mode):
            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += bs

    return total_loss / total_n, total_correct / total_n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--cutout-size", type=int, default=0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--output-dir", type=str, default="./runs/baseline")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device)
    train_loader, test_loader = get_dataloaders(
        root=args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers, cutout_size=args.cutout_size,
        seed=args.seed)

    model = build_model(num_classes=10, width_mult=args.width_mult,
                        dropout=args.dropout).to(device)
    n_params = sum(param.numel() for param in model.parameters())

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    scheduler = build_lr_scheduler(optimizer, args.epochs, args.warmup_epochs,
                                   len(train_loader), args.lr)

    history_path = os.path.join(args.output_dir, "history.csv")
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "test_loss", "test_acc",
             "lr", "epoch_time_s"])

    print(f"model params: {n_params:,}")
    print(f"device: {device}, train batches/epoch: {len(train_loader)}")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                          device, optimizer, scheduler)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
        epoch_time = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, train_loss, train_acc, test_loss, test_acc, cur_lr,
                 epoch_time])

        print(f"epoch {epoch:3d}/{args.epochs} | "
              f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"test_loss {test_loss:.4f} acc {test_acc:.4f} | "
              f"lr {cur_lr:.5f} | {epoch_time:.1f}s")

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "test_acc": test_acc,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.output_dir, "last.pt"))
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))

    print(f"done. best test acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
