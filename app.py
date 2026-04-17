import os
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image


st.set_page_config(page_title="GroundedStyle Demo", layout="wide")
st.title("GroundedStyle: Selective Background Stylization")
st.caption("UI for running main.py (NST) and main2.py (CycleGAN background stylization)")


METHODS = [
    "Vanilla NST",
    "CycleGAN",
]


def save_uploaded_file(uploaded_file, out_path):
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def normalize_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    if prompt and not prompt.endswith("."):
        prompt += "."
    return prompt


# def run_command(cmd):
#     st.code(" ".join(cmd), language="bash")
#     result = subprocess.run(cmd, capture_output=True, text=True)

#     if result.stdout:
#         st.text_area("Stdout", result.stdout, height=250)
#     if result.stderr:
#         st.text_area("Stderr", result.stderr, height=250)

#     if result.returncode != 0:
#         raise RuntimeError(f"Command failed with exit code {result.returncode}")

def run_command(cmd):
    with st.spinner("Processing..."):
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        err_msg = result.stderr if result.stderr else "Unknown error while running the script."
        raise RuntimeError(err_msg)

with st.sidebar:
    st.header("Inputs")
    method = st.selectbox("Choose method", METHODS)

    uploaded_content = st.file_uploader(
        "Upload content image",
        type=["png", "jpg", "jpeg", "webp"]
    )

    uploaded_style = None
    if method == "Vanilla NST":
        uploaded_style = st.file_uploader(
            "Upload style image",
            type=["png", "jpg", "jpeg", "webp"]
        )

    text_prompt = st.text_input("Segmentation prompt", value="a person.")
    image_size = st.slider("Image size", 128, 512, 256, step=32)
    output_dir = st.text_input("Output folder", value="outputs/demo")

    if method == "Vanilla NST":
        steps = st.slider("NST steps", 50, 500, 200, step=25)

    st.divider()
    st.subheader("Grounded SAM settings")
    box_threshold = st.slider("Box threshold", 0.05, 0.9, 0.35, step=0.05)
    text_threshold = st.slider("Text threshold", 0.05, 0.9, 0.25, step=0.05)
    mask_threshold = st.slider("Mask threshold", 0.05, 0.95, 0.5, step=0.05)
    mask_blur = st.slider("Mask blur", 1, 51, 21, step=2)

    st.divider()
    st.subheader("CycleGAN settings")
    cyclegan_repo = st.text_input("CycleGAN repo path", value="pytorch-CycleGAN-and-pix2pix")
    cyclegan_name = st.text_input("Experiment name", value="monet_bg_model")
    cyclegan_epoch = st.text_input("Checkpoint epoch", value="60")


run_btn = st.button("Run demo", type="primary")

if run_btn:
    if uploaded_content is None:
        st.error("Please upload a content image.")
        st.stop()

    if method == "Vanilla NST" and uploaded_style is None:
        st.error("Please upload a style image for Vanilla NST.")
        st.stop()

    os.makedirs(output_dir, exist_ok=True)

    text_prompt = normalize_prompt(text_prompt)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        content_path = tmpdir / uploaded_content.name
        save_uploaded_file(uploaded_content, content_path)

        style_path = None
        if uploaded_style is not None:
            style_path = tmpdir / uploaded_style.name
            save_uploaded_file(uploaded_style, style_path)

        try:
            if method == "Vanilla NST":
                cmd = [
                    sys.executable,
                    "main.py",
                    "--content", str(content_path),
                    "--style", str(style_path),
                    "--output_dir", output_dir,
                    "--image_size", str(image_size),
                    "--steps", str(steps),
                    "--text_prompt", text_prompt,
                    "--box_threshold", str(box_threshold),
                    "--text_threshold", str(text_threshold),
                    "--mask_threshold", str(mask_threshold),
                    "--mask_blur", str(mask_blur),
                ]

                run_command(cmd)

                result_paths = [
                    ("Content", os.path.join(output_dir, "content_resized.png")),
                    ("Style", os.path.join(output_dir, "style_resized.png")),
                    ("Foreground Mask", os.path.join(output_dir, "foreground_mask.png")),
                    ("Vanilla NST", os.path.join(output_dir, "vanilla_nst.png")),
                    ("Masked NST", os.path.join(output_dir, "masked_nst.png")),
                    ("Comparison", os.path.join(output_dir, "comparison.png")),
                ]

            else:  # CycleGAN
                cmd = [
                    sys.executable,
                    "main2.py",
                    "--content", str(content_path),
                    "--output_dir", output_dir,
                    "--image_size", str(image_size),
                    "--text_prompt", text_prompt,
                    "--box_threshold", str(box_threshold),
                    "--text_threshold", str(text_threshold),
                    "--mask_threshold", str(mask_threshold),
                    "--mask_blur", str(mask_blur),
                    "--cyclegan_repo", cyclegan_repo,
                    "--cyclegan_name", cyclegan_name,
                    "--cyclegan_epoch", cyclegan_epoch,
                ]

                run_command(cmd)

                result_paths = [
                    ("Content", os.path.join(output_dir, "content_resized.png")),
                    ("Foreground Mask (raw)", os.path.join(output_dir, "foreground_mask_raw.png")),
                    ("Foreground Mask (visual)", os.path.join(output_dir, "foreground_mask_vis.png")),
                    ("CycleGAN Full", os.path.join(output_dir, "cyclegan_full.png")),
                    ("CycleGAN Background Only", os.path.join(output_dir, "cyclegan_background_only.png")),
                    ("Comparison", os.path.join(output_dir, "comparison.png")),
                ]

            st.success(f"Saved outputs to: {output_dir}")

            for title, path in result_paths:
                if os.path.exists(path):
                    st.image(Image.open(path), caption=title, use_container_width=True)

        except Exception as e:
            st.error(str(e))