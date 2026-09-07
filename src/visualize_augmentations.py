"""
Visualize CIFAR-10 augmentations from Q1(a) — show original + 4 augmented variants
of each image to illustrate the effect of RandomCrop(reflect), RandomHorizontalFlip,
and Normalize.

    python src/visualize_augmentations.py --output-dir ./runs/augmented_data_q1
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from data import CLASSES, CIFAR10_MEAN, CIFAR10_STD


def denormalize(x):
    """Reverse the normalization (x is in [-1, 1] after Normalize(mean, std))."""
    x = x.clone()
    for i in range(3):
        x[i] = x[i] * CIFAR10_STD[i] + CIFAR10_MEAN[i]
    return x.clamp(0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default="./runs/augmented_data_q1")
    p.add_argument("--num-samples", type=int, default=5,
                   help="number of unique images to visualize")
    p.add_argument("--num-augmentations", type=int, default=5,
                   help="number of augmented variants per image (including original)")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset without any transforms to get raw images
    dataset = datasets.CIFAR10(args.data_root, train=True, download=False,
                               transform=transforms.ToTensor())

    # The augmentation pipeline used during training
    augment_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    # Normalize (will denormalize for display)
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

    # Pick random sample indices
    indices = torch.randperm(len(dataset))[:args.num_samples]

    for sample_idx, idx in enumerate(indices):
        img, label = dataset[idx]  # img is already ToTensor'd, so [0,1]

        # Create a figure with 1 + num_augmentations columns
        fig, axes = plt.subplots(1, args.num_augmentations, figsize=(15, 3))
        if args.num_augmentations == 1:
            axes = [axes]

        # First column: original image (with normalization applied, then denormalized for display)
        img_norm = normalize(img.clone())
        img_denorm = denormalize(img_norm)
        ax = axes[0]
        ax.imshow(img_denorm.permute(1, 2, 0).numpy())
        ax.set_title(f"Original\n{CLASSES[label]}", fontsize=10)
        ax.axis("off")

        # Remaining columns: augmented versions
        for aug_idx in range(1, args.num_augmentations):
            # Apply augmentations to the original (ToTensor'd) image
            img_pil = transforms.ToPILImage()(img)
            img_aug = augment_transform(img_pil)  # returns a PIL image
            img_aug_tensor = transforms.ToTensor()(img_aug)

            # Normalize and denormalize for display
            img_aug_norm = normalize(img_aug_tensor.clone())
            img_aug_denorm = denormalize(img_aug_norm)

            ax = axes[aug_idx]
            ax.imshow(img_aug_denorm.permute(1, 2, 0).numpy())
            ax.set_title(f"Augmented {aug_idx}\n{CLASSES[label]}", fontsize=10)
            ax.axis("off")

        fig.suptitle(f"Sample {sample_idx + 1}: {CLASSES[label]} "
                     f"(RandomCrop + RandomFlip variations)", fontsize=12)
        fig.tight_layout()

        out_path = os.path.join(args.output_dir, f"augmented_{sample_idx:02d}_{CLASSES[label]}.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path}")

    print(f"\nAll augmented samples saved to {args.output_dir}")


if __name__ == "__main__":
    main()
