import argparse
import os
import torch

from masked_nst import run_masked_nst
from utils.image_utils import (
    ensure_same_size,
    get_device,
    load_image_tensor,
    save_tensor_image,
)
from utils.visualization import save_mask_visualization, save_side_by_side
from vanilla_nst import run_vanilla_nst
from models.grounded_sam_mask import GroundedSAMMaskGenerator

# YOLO imports commented out for now
# from models.yolo_segmentation import (
#     generate_foreground_mask_yolo,
#     load_yolo_segmentation_model,
#     print_available_classes,
# )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grounded SAM Guided Neural Style Transfer"
    )
    parser.add_argument("--content", type=str, required=True, help="Path to content image")
    parser.add_argument("--style", type=str, required=True, help="Path to style image")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--image_size", type=int, default=256, help="NST image size")
    parser.add_argument("--steps", type=int, default=250, help="NST optimization steps")

    # Mask / blending options
    parser.add_argument("--mask_threshold", type=float, default=0.5, help="Mask threshold")
    parser.add_argument("--mask_blur", type=int, default=21, help="Mask blur kernel")

    # Grounded SAM options
    parser.add_argument(
        "--text_prompt",
        type=str,
        required=True,
        help='Text prompt for Grounding DINO, e.g. "a bridge."',
    )
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument(
        "--grounding_model",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
        help="Grounding DINO model id",
    )
    parser.add_argument(
        "--sam_model",
        type=str,
        default="facebook/sam-vit-base",
        help="SAM model id",
    )

    # YOLO args commented out for now
    # parser.add_argument("--yolo_model", type=str, default="yolov8n-seg.pt", help="YOLO segmentation model")
    # parser.add_argument(
    #     "--target_class",
    #     type=str,
    #     default=None,
    #     help="Object class to preserve, e.g. person, car, dog, bus."
    # )
    # parser.add_argument("--yolo_imgsz", type=int, default=640, help="YOLO inference size")
    # parser.add_argument("--list_classes", action="store_true", help="Print YOLO classes and exit")
    # parser.add_argument("--use_grounded_sam", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    print(f"Using NST device: {device}")

    # 1. Load content/style for NST
    content_img = load_image_tensor(args.content, args.image_size, device)
    style_img = load_image_tensor(args.style, args.image_size, device)
    content_img, style_img = ensure_same_size(content_img, style_img)

    save_tensor_image(content_img, os.path.join(args.output_dir, "content_resized.png"))
    save_tensor_image(style_img, os.path.join(args.output_dir, "style_resized.png"))

    # 2. Generate foreground mask using Grounding DINO + SAM
    grounded_sam = GroundedSAMMaskGenerator(
        device=device,
        grounding_model_id=args.grounding_model,
        sam_model_id=args.sam_model,
    )

    fg_mask, selected_label, selected_score, boxes, labels, scores = grounded_sam.generate_mask(
        image_path=args.content,
        text_prompt=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        blur_kernel=args.mask_blur,
        threshold=args.mask_threshold,
        select="largest",
    )

    # Resize mask to NST image size
    fg_mask = fg_mask.to(device)
    fg_mask = torch.nn.functional.interpolate(
        fg_mask,
        size=content_img.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(0, 1)

    print(
        f"Selected foreground object: label='{selected_label}', "
        f"score={selected_score:.4f}"
    )
    save_mask_visualization(fg_mask, os.path.join(args.output_dir, "foreground_mask.png"))

    # YOLO path commented out for now
    # yolo_model = load_yolo_segmentation_model(args.yolo_model)
    #
    # if args.list_classes:
    #     print_available_classes(yolo_model)
    #     return
    #
    # fg_mask, selected_class, selected_conf = generate_foreground_mask_yolo(
    #     model=yolo_model,
    #     image_path=args.content,
    #     image_size=args.yolo_imgsz,
    #     target_class=args.target_class,
    #     threshold=args.mask_threshold,
    #     blur_kernel=args.mask_blur,
    #     retina_masks=True,
    #     verbose=False,
    # )
    #
    # fg_mask = fg_mask.to(device)
    # fg_mask = torch.nn.functional.interpolate(
    #     fg_mask,
    #     size=content_img.shape[-2:],
    #     mode="bilinear",
    #     align_corners=False,
    # ).clamp(0, 1)
    #
    # print(f"Selected foreground object: class='{selected_class}', confidence={selected_conf:.4f}")
    # save_mask_visualization(fg_mask, os.path.join(args.output_dir, "foreground_mask.png"))

    # 3. Vanilla baseline
    vanilla_result = run_vanilla_nst(
        content_img=content_img,
        style_img=style_img,
        num_steps=args.steps,
        style_weight=1e5,
        content_weight=1.0,
        tv_weight=1e-5,
        lr=0.02,
    )
    save_tensor_image(vanilla_result, os.path.join(args.output_dir, "vanilla_nst.png"))

    # 4. Grounded-SAM-guided masked NST
    masked_result = run_masked_nst(
        content_img=content_img,
        style_img=style_img,
        fg_mask=fg_mask,
        num_steps=args.steps,
        fg_content_weight=3.0,
        global_content_weight=0.2,
        bg_style_weight=1e5,
        tv_weight=1e-5,
        lr=0.02,
    )
    save_tensor_image(masked_result, os.path.join(args.output_dir, "masked_nst.png"))

    # 5. Comparison panel
    save_side_by_side(
        images=[content_img, style_img, fg_mask, vanilla_result, masked_result],
        titles=["Content", "Style", "FG Mask", "Vanilla NST", "Grounded SAM NST"],
        path=os.path.join(args.output_dir, "comparison.png"),
    )

    print("\nDone.")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()