# Selective NST & CycleGAN via Grounded Segmentation

This project contains two separate pipelines:

- `main.py`  
  Vanilla NST + masked NST using Grounded SAM foreground masks.

- `main2.py`  
  Segmentation + trained CycleGAN pipeline for **background-only stylization**.

---

## Project structure

```text
NST/
├── main.py
├── main2.py
├── vanilla_nst.py
├── masked_nst.py
├── models/
├── utils/
├── data/
│   ├── content/
│   └── style/
├── outputs/
└── pytorch-CycleGAN-and-pix2pix/
```

The folder `cyclegan_train` contains the CycleGAN training code.

---

## Setup

Create and activate the virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the main dependencies:

```bash
pip install torch torchvision pillow scikit-image dominate transformers accelerate timm
```

If you are using Grounded SAM through Hugging Face models, you may also want:

```bash
pip install sentencepiece safetensors
```

If anything is missing during runtime, install that package in the same `.venv`.

---

## 1. Run `main.py` (Vanilla NST + Masked NST)

This pipeline:
- loads the content image
- loads the style image
- generates a foreground mask using Grounded SAM
- runs vanilla NST on the full image
- runs masked NST so style is focused on the background
- saves a comparison panel

### Command

```bash
python main.py \
  --content data/content/person2.jpg \
  --style data/style/the_scream.jpg \
  --output_dir outputs/groundedDINO2 \
  --image_size 256 \
  --steps 200 \
  --text_prompt "a person."
```

### Outputs

This saves files like:
- `content_resized.png`
- `style_resized.png`
- `foreground_mask.png`
- `vanilla_nst.png`
- `masked_nst.png`
- `comparison.png`

---

## 2. Run `main2.py` (Segmentation + CycleGAN background-only stylization)

This pipeline:
- runs segmentation first
- saves the foreground mask
- runs the trained CycleGAN model on the full image
- reloads the saved mask
- blends using:

```text
final = I * M + IB * (1 - M)
```

where:
- `I` = original image
- `IB` = CycleGAN stylized full image
- `M` = foreground mask (white foreground, black background)

So the foreground remains original and the background gets stylized.

### Command

```bash
python main2.py \
  --content data/content/person2.jpg \
  --output_dir outputs/gan_bg_run1 \
  --image_size 256 \
  --text_prompt "a person." \
  --cyclegan_repo pytorch-CycleGAN-and-pix2pix \
  --cyclegan_name monet_bg_model \
  --cyclegan_epoch 60
```

### Outputs

This saves files like:
- `content_resized.png`
- `foreground_mask_raw.png`
- `foreground_mask_vis.png`
- `cyclegan_full.png`
- `cyclegan_background_only.png`
- `comparison.png`

---

## Notes on `main2.py`

### Important

The CycleGAN repo must already exist locally:

```text
pytorch-CycleGAN-and-pix2pix/
```

and the trained checkpoint files must be present inside:

```text
pytorch-CycleGAN-and-pix2pix/checkpoints/monet_bg_model/
```

For example:

```text
60_net_G_A.pth
60_net_G_B.pth
```

or:

```text
latest_net_G_A.pth
latest_net_G_B.pth
```

### Prompt formatting

For Grounded SAM / Grounding DINO, prompts often work better with a trailing period.

Example:

```bash
--text_prompt "a person."
```

instead of:

```bash
--text_prompt "a person"
```

### Temporary CycleGAN inference dataset

If `main2.py` calls `test.py` from the CycleGAN repo using `dataset_mode=unaligned`, then both of these temporary folders must contain at least one image:

```text
testA/
testB/
```

Even if you only care about `A -> B` translation, `testB` must not be empty.
A simple fix is to copy the same content image into both `testA` and `testB`.

---

## Example content/style files

You can place your images like this:

```text
data/content/person2.jpg
data/style/the_scream.jpg
```

---

## Summary

### `main.py`
Use this when you want:
- vanilla NST baseline
- masked NST comparison
- style image input

### `main2.py`
Use this when you want:
- segmentation mask first
- pretrained GAN stylization
- background-only blending using the mask
- no vanilla NST in this pipeline

---
