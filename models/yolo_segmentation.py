from typing import Optional, Tuple

import torch
from ultralytics import YOLO

from utils.mask_utils import refine_mask


def load_yolo_segmentation_model(model_name: str = "yolov8n-seg.pt"):
    """
    Loads a pretrained Ultralytics YOLO segmentation model.
    Good starter options:
    - yolov8n-seg.pt  (fastest)
    - yolov8s-seg.pt  (better quality)
    """
    model = YOLO(model_name)
    return model


def get_class_name_map(model) -> dict:
    """
    Returns class index -> class name mapping from YOLO model.
    """
    return model.names


def print_available_classes(model) -> None:
    names = model.names
    print("\nAvailable YOLO classes:")
    for idx, name in names.items():
        print(f"{idx}: {name}")


def select_best_instance(
    result,
    target_class: Optional[str] = None,
) -> Optional[int]:
    """
    Selects one instance index from YOLO result.
    If target_class is given, picks the highest-confidence instance of that class.
    Otherwise picks the highest-confidence instance overall.
    """
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes
    cls_ids = boxes.cls.detach().cpu().tolist()
    confs = boxes.conf.detach().cpu().tolist()

    best_idx = None
    best_conf = -1.0

    for i, (cls_id, conf) in enumerate(zip(cls_ids, confs)):
        class_name = result.names[int(cls_id)]

        if target_class is not None and class_name != target_class:
            continue

        if conf > best_conf:
            best_conf = conf
            best_idx = i

    return best_idx


def generate_foreground_mask_yolo(
    model,
    image_path: str,
    image_size: int = 640,
    target_class: Optional[str] = None,
    threshold: float = 0.5,
    blur_kernel: int = 21,
    retina_masks: bool = True,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Optional[str], Optional[float]]:
    """
    Runs YOLO segmentation on image_path and returns:
    - fg_mask: [1,1,H,W] float tensor in [0,1]
    - selected_class_name
    - selected_confidence

    If target_class is None, the highest-confidence segmented object is used.
    If target_class is provided (e.g. "car", "dog", "person"), the best instance
    of that class is used.
    """
    results = model.predict(
        source=image_path,
        imgsz=image_size,
        retina_masks=retina_masks,
        verbose=verbose,
    )

    if len(results) == 0:
        raise RuntimeError("YOLO returned no results.")

    result = results[0]

    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError("No segmented objects were found in the image.")

    selected_idx = select_best_instance(result, target_class=target_class)
    if selected_idx is None:
        if target_class is None:
            raise RuntimeError("No suitable segmented instance found.")
        raise RuntimeError(f"No instance found for target_class='{target_class}'")

    # masks.data shape: [N, H, W]
    mask = result.masks.data[selected_idx].float().unsqueeze(0).unsqueeze(0)

    cls_id = int(result.boxes.cls[selected_idx].item())
    conf = float(result.boxes.conf[selected_idx].item())
    class_name = result.names[cls_id]

    # refine to make soft edges for cleaner blending
    mask = refine_mask(mask, threshold=threshold, blur_kernel=blur_kernel)

    return mask, class_name, conf