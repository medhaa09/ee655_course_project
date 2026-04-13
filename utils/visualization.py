import os
import matplotlib.pyplot as plt
import torch


def save_mask_visualization(mask: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mask_np = mask.detach().cpu().squeeze().numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(mask_np, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_side_by_side(images, titles, path: str) -> None:
    """
    images: list of tensors [1,3,H,W] or [1,1,H,W]
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(images)
    plt.figure(figsize=(4 * n, 4))

    for i, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, n, i)
        arr = img.detach().cpu().squeeze()
        if arr.dim() == 2:
            plt.imshow(arr.numpy(), cmap="gray")
        else:
            plt.imshow(arr.permute(1, 2, 0).clamp(0, 1).numpy())
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()