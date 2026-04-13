import argparse
import os

import torch

from masked_nst import run_masked_nst
from models.segmentation import load_segmentation_model, generate_foreground_mask
from utils.image_utils import (
    ensure_same_size,
    get_device,
    load_image_tensor,
    save_tensor_image,
)
from utils.visualization import save_mask_visualization, save_side_by_side
from vanilla_nst import run_vanilla_nst


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic-Aware Neural Style Transfer")
    parser.add_argument("--content", type=str, required=True, help="Path to content image")
    parser.add_argument("--style", type=str, required=True, help="Path to style image")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--image_size", type=int, default=256, help="Resize size")
    parser.add_argument("--steps", type=int, default=300, help="Optimization steps")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    content_img = load_image_tensor(args.content, args.image_size, device)
    style_img = load_image_tensor(args.style, args.image_size, device)
    content_img, style_img = ensure_same_size(content_img, style_img)

    # Save resized inputs
    save_tensor_image(content_img, os.path.join(args.output_dir, "content_resized.png"))
    save_tensor_image(style_img, os.path.join(args.output_dir, "style_resized.png"))

    # 1. Foreground mask
    seg_model = load_segmentation_model(device)
    fg_mask = generate_foreground_mask(seg_model, content_img, target_class_idx=15, threshold=0.25, blur_kernel=21)

    save_mask_visualization(fg_mask, os.path.join(args.output_dir, "foreground_mask.png"))

    # 2. Vanilla NST baseline
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

    # 3. Masked NST
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

    # 4. Comparison figure
    save_side_by_side(
        images=[content_img, style_img, fg_mask, vanilla_result, masked_result],
        titles=["Content", "Style", "FG Mask", "Vanilla NST", "Masked NST"],
        path=os.path.join(args.output_dir, "comparison.png"),
    )

    print("\nDone.")
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()