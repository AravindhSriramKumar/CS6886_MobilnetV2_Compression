"""
CIFAR-10 data pipeline for CS6886 Assignment 2 (Q1a).

Provides the train/test transforms and DataLoaders used by every other stage
(training, evaluation, calibration for quantization).  Normalization statistics
are the channel-wise mean/std computed over the 50k CIFAR-10 training images.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Channel-wise statistics of the CIFAR-10 training split (RGB, [0,1] scale).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


def build_transforms(img_size=32, augment=True, cutout_size=0):
    """Return (train_transform, test_transform).

    Train-time augmentation (applied on-the-fly, per epoch):
      * RandomCrop(32, padding=4, padding_mode='reflect') -- translation
        invariance; reflect padding avoids the black border bias of zero pad.
      * RandomHorizontalFlip(p=0.5) -- CIFAR-10 classes are left/right
        symmetric, so this is label preserving (unlike vertical flips).
      * ToTensor + Normalize -- puts every channel at ~zero mean/unit variance,
        which keeps BatchNorm statistics well conditioned.
      * Optional Cutout / RandomErasing of a `cutout_size` square -- extra
        occlusion regularization for the longer MobileNet-v2 schedules.

    Test-time: deterministic ToTensor + the *same* Normalize only. No random
    ops, so evaluation numbers are reproducible.

    `img_size` > 32 additionally resizes (used only if MobileNet-v2 ImageNet
    pretrained weights at 224x224 are exercised).
    """
    train_ops = []
    test_ops = []

    if img_size != 32:
        train_ops.append(transforms.Resize(img_size))
        test_ops.append(transforms.Resize(img_size))

    if augment:
        train_ops += [
            transforms.RandomCrop(img_size, padding=img_size // 8,
                                  padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
        ]

    norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_ops += [transforms.ToTensor(), norm]
    test_ops += [transforms.ToTensor(), norm]

    if augment and cutout_size > 0:
        # RandomErasing operates on tensors, so it comes after Normalize.
        # value=0 in normalized space == the dataset mean pixel.
        train_ops.append(transforms.RandomErasing(
            p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0.0))

    return transforms.Compose(train_ops), transforms.Compose(test_ops)


def get_datasets(root="./data", img_size=32, augment=True, cutout_size=0,
                 download=True):
    train_tf, test_tf = build_transforms(img_size, augment, cutout_size)
    train_set = datasets.CIFAR10(root, train=True, download=download,
                                 transform=train_tf)
    test_set = datasets.CIFAR10(root, train=False, download=download,
                                transform=test_tf)
    return train_set, test_set


def get_dataloaders(root="./data", batch_size=128, num_workers=4, img_size=32,
                    augment=True, cutout_size=0, seed=42, download=True):
    """DataLoaders with a seeded generator so shuffling is reproducible."""
    train_set, test_set = get_datasets(root, img_size, augment, cutout_size,
                                       download)
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=num_workers > 0, generator=g,
        worker_init_fn=_worker_init_fn)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0)
    return train_loader, test_loader


def _worker_init_fn(worker_id):
    """Give each dataloader worker a distinct but deterministic seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)


def compute_dataset_stats(root="./data"):
    """Recompute the CIFAR-10 train mean/std (used to justify the constants)."""
    ds = datasets.CIFAR10(root, train=True, download=True,
                          transform=transforms.ToTensor())
    x = torch.stack([ds[i][0] for i in range(len(ds))])   # N,3,32,32
    return x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3))


if __name__ == "__main__":
    train_tf, test_tf = build_transforms()
    print("TRAIN:", train_tf)
    print("TEST :", test_tf)
    mean, std = compute_dataset_stats()
    print("recomputed mean:", [round(v, 4) for v in mean.tolist()])
    print("recomputed std :", [round(v, 4) for v in std.tolist()])
    tr, te = get_dataloaders(num_workers=2)
    xb, yb = next(iter(tr))
    print("batch:", tuple(xb.shape), "per-channel batch mean:",
          [round(v, 3) for v in xb.mean(dim=(0, 2, 3)).tolist()])
    print("train/test sizes:", len(tr.dataset), len(te.dataset))
