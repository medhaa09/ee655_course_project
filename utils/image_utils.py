import os
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms


# ImageNet normalization for VGG
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pil_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def build_image_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def load_image_tensor(path: str, image_size: int, device: torch.device) -> torch.Tensor:
    """
    Returns tensor of shape [1, 3, H, W] in range [0, 1].
    """
    image = load_pil_image(path)
    transform = build_image_transform(image_size)
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    tensor: [1,3,H,W] or [3,H,W], assumed in [0,1]
    """
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    return transforms.ToPILImage()(tensor)


def save_tensor_image(tensor: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = tensor_to_pil(tensor)
    image.save(path)


def normalize_for_vgg(img: torch.Tensor) -> torch.Tensor:
    """
    img in [0,1], shape [B,3,H,W]
    """
    mean = torch.tensor(IMAGENET_MEAN, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=img.device).view(1, 3, 1, 1)
    return (img - mean) / std


def denormalize_from_vgg(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=img.device).view(1, 3, 1, 1)
    return img * std + mean


def ensure_same_size(content: torch.Tensor, style: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Resize style to content spatial size if needed.
    """
    if content.shape[-2:] != style.shape[-2:]:
        style = torch.nn.functional.interpolate(
            style,
            size=content.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return content, style