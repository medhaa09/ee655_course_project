from typing import Optional

import torch
import torch.nn.functional as F
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

from utils.mask_utils import refine_mask


# VOC-style class list used by these weights:
# 0 background, 15 person, etc.
PERSON_CLASS_INDEX = 15


def load_segmentation_model(device: torch.device) -> torch.nn.Module:
    """
    Loads pretrained DeepLabV3-ResNet50.
    """
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model


def generate_foreground_mask(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class_idx: int = PERSON_CLASS_INDEX,
    threshold: float = 0.5,
    blur_kernel: int = 21,
) -> torch.Tensor:
    """
    image_tensor: [1,3,H,W] in [0,1]
    Returns soft mask [1,1,H,W] where foreground ~ 1.
    """
    with torch.no_grad():
        output = model(image_tensor)["out"]  # [1, C, H, W]
        probs = F.softmax(output, dim=1)     # [1, C, H, W]
        class_prob = probs[:, target_class_idx:target_class_idx + 1, :, :]  # [1,1,H,W]

    mask = refine_mask(class_prob, threshold=threshold, blur_kernel=blur_kernel)
    return mask


def generate_binary_salient_mask_from_prediction(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    preferred_class_idx: Optional[int] = PERSON_CLASS_INDEX,
) -> torch.Tensor:
    """
    Optional helper if you later want to fallback to argmax segmentation.
    """
    with torch.no_grad():
        logits = model(image_tensor)["out"]
        pred = torch.argmax(logits, dim=1, keepdim=True).float()

    if preferred_class_idx is not None:
        mask = (pred == float(preferred_class_idx)).float()
    else:
        mask = (pred > 0).float()

    return mask