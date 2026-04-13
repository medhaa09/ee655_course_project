import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    SamModel,
    SamProcessor,
)

from utils.mask_utils import refine_mask


class GroundedSAMMaskGenerator:
    def __init__(
        self,
        device: torch.device,
        grounding_model_id: str = "IDEA-Research/grounding-dino-tiny",
        sam_model_id: str = "facebook/sam-vit-base",
    ):
        self.device = device

        # Grounding DINO
        self.grounding_processor = AutoProcessor.from_pretrained(grounding_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_model_id
        ).to(device)
        self.grounding_model.eval()

        # SAM
        self.sam_processor = SamProcessor.from_pretrained(sam_model_id)
        self.sam_model = SamModel.from_pretrained(sam_model_id).to(device)
        self.sam_model.eval()

    @staticmethod
    def _load_pil(image_path: str) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def detect_boxes(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        """
        Returns detected boxes and labels from Grounding DINO.

        text_prompt examples:
        - "a bridge."
        - "a tower."
        - "a bridge. a tower."
        """
        inputs = self.grounding_processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        # Handle both older and newer Transformers signatures
        try:
            results = self.grounding_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],  # (H, W)
            )
        except TypeError:
            results = self.grounding_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],  # (H, W)
            )

        return results[0]

    def segment_from_boxes(
        self,
        image: Image.Image,
        boxes_xyxy: torch.Tensor,
        multimask_output: bool = False,
    ):
        """
        Takes Grounding DINO boxes and returns SAM masks.
        boxes_xyxy: tensor [N, 4] in xyxy pixel coordinates
        """
        if boxes_xyxy.numel() == 0:
            raise RuntimeError("No boxes available for SAM prompting.")

        # SAM expects boxes in nested list format for the processor
        input_boxes = [boxes_xyxy.cpu().tolist()]

        inputs = self.sam_processor(
            image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.sam_model(**inputs, multimask_output=multimask_output)

        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )

        iou_scores = outputs.iou_scores.detach().cpu()
        return masks, iou_scores

    def generate_mask(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        blur_kernel: int = 21,
        threshold: float = 0.5,
        select: str = "largest",
    ):
        """
        Full pipeline:
        image + text -> Grounding DINO boxes -> SAM masks -> single foreground mask

        select:
        - "largest": choose largest detected mask
        - "best_box": choose highest-confidence box
        """
        image = self._load_pil(image_path)

        detection = self.detect_boxes(
            image=image,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        boxes = detection["boxes"]    # [N, 4]
        scores = detection["scores"]  # [N]
        labels = detection["labels"]  # list[str]

        if boxes is None or len(boxes) == 0:
            raise RuntimeError(f"No object detected for prompt: {text_prompt}")

        masks, iou_scores = self.segment_from_boxes(image, boxes)

        # Usually masks[0] corresponds to the batch item
        mask_batch = masks[0]

        if isinstance(mask_batch, list):
            mask_batch = torch.stack(mask_batch)

        # Expected shapes after post-processing can vary by version:
        # [N, H, W] or [N, 1, H, W]
        if mask_batch.dim() == 3:
            mask_batch = mask_batch.unsqueeze(1)  # [N,1,H,W]
        elif mask_batch.dim() == 4:
            pass
        else:
            raise RuntimeError(f"Unexpected SAM mask shape: {mask_batch.shape}")

        mask_batch = mask_batch.float()

        if select == "best_box":
            selected_idx = int(torch.argmax(scores).item())
        else:
            # choose largest mask area
            areas = mask_batch[:, 0].flatten(1).sum(dim=1)
            selected_idx = int(torch.argmax(areas).item())

        chosen_mask = mask_batch[selected_idx:selected_idx + 1]  # [1,1,H,W]
        chosen_label = labels[selected_idx]
        chosen_score = float(scores[selected_idx].item())

        chosen_mask = refine_mask(
            chosen_mask,
            threshold=threshold,
            blur_kernel=blur_kernel,
        )

        return chosen_mask, chosen_label, chosen_score, boxes, labels, scores