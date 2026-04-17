import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

from utils.image_utils import (
    ensure_same_size,
    get_device,
    load_image_tensor,
    save_tensor_image,
)
from utils.visualization import save_mask_visualization, save_side_by_side
from models.grounded_sam_mask import GroundedSAMMaskGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segmentation + CycleGAN background-only stylization pipeline"
    )

    parser.add_argument("--content", type=str, required=True, help="Path to content image")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--image_size", type=int, default=256, help="Working image size")

    # Mask options
    parser.add_argument("--mask_threshold", type=float, default=0.5, help="Mask threshold")
    parser.add_argument("--mask_blur", type=int, default=21, help="Mask blur kernel")

    # Grounded SAM options
    parser.add_argument(
        "--text_prompt",
        type=str,
        required=True,
        help='Text prompt for Grounding DINO, e.g. "a person"',
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

    # CycleGAN options
    parser.add_argument(
        "--cyclegan_repo",
        type=str,
        required=True,
        help="Path to local pytorch-CycleGAN-and-pix2pix repo",
    )
    parser.add_argument(
        "--cyclegan_name",
        type=str,
        default="monet_bg_model",
        help="Checkpoint folder name inside repo/checkpoints/",
    )
    parser.add_argument(
        "--cyclegan_epoch",
        type=str,
        default="latest",
        help="Checkpoint epoch to use, e.g. latest or 60",
    )

    return parser.parse_args()


def run_command(cmd, cwd=None):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def save_raw_mask(mask_tensor, path):
    """
    Save mask as grayscale PNG:
    white = foreground, black = background
    """
    mask = mask_tensor.detach().cpu().squeeze().clamp(0, 1).numpy()
    mask_uint8 = (mask * 255).astype(np.uint8)
    Image.fromarray(mask_uint8, mode="L").save(path)


def load_raw_mask(mask_path, image_size, device):
    """
    Load saved grayscale mask and return tensor of shape [1,1,H,W] in [0,1].
    """
    mask = Image.open(mask_path).convert("L").resize((image_size, image_size))
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
    return mask.clamp(0, 1)


def find_fake_b_image(results_dir):
    candidates = sorted(glob.glob(os.path.join(results_dir, "*_fake_B.*")))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No fake_B image found in {results_dir}")
    return candidates[0]


def run_cyclegan_on_single_image(content_path, output_dir, repo_dir, model_name, epoch):
    """
    Run CycleGAN test.py on one content image and return the saved fake_B path.
    """
    repo_dir = os.path.abspath(repo_dir)
    output_dir = os.path.abspath(output_dir)
    content_path = os.path.abspath(content_path)

    if not os.path.isdir(repo_dir):
        raise FileNotFoundError(f"CycleGAN repo not found: {repo_dir}")

    test_py = os.path.join(repo_dir, "test.py")
    if not os.path.exists(test_py):
        raise FileNotFoundError(f"test.py not found inside repo: {test_py}")

    with tempfile.TemporaryDirectory(dir=output_dir) as tmpdir:
        tmpdir = os.path.abspath(tmpdir)
        data_root = os.path.abspath(os.path.join(tmpdir, "cyclegan_input"))
        testA_dir = os.path.join(data_root, "testA")
        testB_dir = os.path.join(data_root, "testB")

        os.makedirs(testA_dir, exist_ok=True)
        os.makedirs(testB_dir, exist_ok=True)

        input_name = "input.png"

        copied_input_A = os.path.join(testA_dir, input_name)
        copied_input_B = os.path.join(testB_dir, input_name)

        shutil.copy2(content_path, copied_input_A)
        shutil.copy2(content_path, copied_input_B)

        cmd = [
            sys.executable,
            "test.py",
            "--dataroot",
            data_root,
            "--name",
            model_name,
            "--model",
            "cycle_gan",
            "--phase",
            "test",
            "--epoch",
            str(epoch),
            "--no_dropout",
        ]

        run_command(cmd, cwd=repo_dir)

        results_dir = os.path.join(
            repo_dir,
            "results",
            model_name,
            f"test_{epoch}",
            "images",
        )

        fake_b_path = find_fake_b_image(results_dir)

        saved_out = os.path.join(output_dir, "cyclegan_full.png")
        shutil.copy2(fake_b_path, saved_out)

    return saved_out
    
def main():
    args = parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # STEP 1: Load original image
    # ---------------------------------------------------------
    content_img = load_image_tensor(args.content, args.image_size, device)
    save_tensor_image(content_img, os.path.join(args.output_dir, "content_resized.png"))

    # ---------------------------------------------------------
    # STEP 2: Run segmentation and save mask first
    # ---------------------------------------------------------
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

    # Save mask in two forms:
    # 1) raw mask used for blending
    # 2) visualization for easy viewing
    raw_mask_path = os.path.join(args.output_dir, "foreground_mask_raw.png")
    mask_vis_path = os.path.join(args.output_dir, "foreground_mask_vis.png")

    save_raw_mask(fg_mask, raw_mask_path)
    save_mask_visualization(fg_mask, mask_vis_path)

    # ---------------------------------------------------------
    # STEP 3: Run CycleGAN full-image stylization
    # ---------------------------------------------------------
    cyclegan_full_path = run_cyclegan_on_single_image(
        content_path=args.content,
        output_dir=args.output_dir,
        repo_dir=args.cyclegan_repo,
        model_name=args.cyclegan_name,
        epoch=args.cyclegan_epoch,
    )

    cyclegan_full = load_image_tensor(cyclegan_full_path, args.image_size, device)
    content_img, cyclegan_full = ensure_same_size(content_img, cyclegan_full)
    save_tensor_image(cyclegan_full, os.path.join(args.output_dir, "cyclegan_full.png"))

    # ---------------------------------------------------------
    # STEP 4: Reload saved mask from disk
    # ---------------------------------------------------------
    fg_mask_saved = load_raw_mask(raw_mask_path, args.image_size, device)
    fg_mask_saved = torch.nn.functional.interpolate(
        fg_mask_saved,
        size=content_img.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(0, 1)

    # ---------------------------------------------------------
    # STEP 5: Blend
    # foreground stays original
    # background becomes GAN-stylized
    # final = I*M + IB*(1-M)
    # ---------------------------------------------------------
    bg_mask = 1.0 - fg_mask_saved
    final_result = content_img * fg_mask_saved + cyclegan_full * bg_mask
    final_result = final_result.clamp(0, 1)

    save_tensor_image(final_result, os.path.join(args.output_dir, "cyclegan_background_only.png"))

    # ---------------------------------------------------------
    # STEP 6: Save comparison
    # ---------------------------------------------------------
    mask_rgb = fg_mask_saved.repeat(1, 3, 1, 1)

    save_side_by_side(
        images=[content_img, mask_rgb, cyclegan_full, final_result],
        titles=["Content", "Saved FG Mask", "CycleGAN Full", "BG-Only Final"],
        path=os.path.join(args.output_dir, "comparison.png"),
    )

    print("\nDone.")
    print(f"Saved outputs to: {args.output_dir}")
    print(f"Raw mask: {raw_mask_path}")
    print(f"CycleGAN full stylized image: {os.path.join(args.output_dir, 'cyclegan_full.png')}")
    print(f"Final background-only stylized image: {os.path.join(args.output_dir, 'cyclegan_background_only.png')}")


if __name__ == "__main__":
    main()