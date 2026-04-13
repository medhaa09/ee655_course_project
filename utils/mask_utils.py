import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


def normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Ensures mask is in [0,1], shape [1,1,H,W].
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    return mask.clamp(0.0, 1.0)


def threshold_mask(mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    mask = normalize_mask(mask)
    return (mask > threshold).float()


def blur_mask(mask: torch.Tensor, kernel_size: int = 21) -> torch.Tensor:
    """
    Applies gaussian blur to soften edges.
    """
    mask = normalize_mask(mask)
    blurred = gaussian_blur(mask.squeeze(0), kernel_size=[kernel_size, kernel_size])
    return blurred.unsqueeze(0).clamp(0.0, 1.0)


def resize_mask(mask: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    mask = normalize_mask(mask)
    return F.interpolate(mask, size=(target_h, target_w), mode="bilinear", align_corners=False)


def invert_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = normalize_mask(mask)
    return 1.0 - mask


def refine_mask(mask: torch.Tensor, threshold: float = 0.5, blur_kernel: int = 21) -> torch.Tensor:
    """
    Binary threshold followed by blur for soft edges.
    """
    mask = threshold_mask(mask, threshold=threshold)
    mask = blur_mask(mask, kernel_size=blur_kernel)
    return mask.clamp(0.0, 1.0)