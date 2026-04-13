from typing import Dict

import torch
from torch.optim import Adam

from models.nst_losses import content_loss, style_loss, total_variation_loss
from models.vgg_features import VGGFeatureExtractor
from utils.image_utils import normalize_for_vgg


STYLE_LAYERS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
CONTENT_LAYER = "conv4_2"


def run_vanilla_nst(
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    num_steps: int = 300,
    style_weight: float = 1e5,
    content_weight: float = 1.0,
    tv_weight: float = 1e-5,
    lr: float = 0.02,
    log_every: int = 50,
) -> torch.Tensor:
    device = content_img.device
    extractor = VGGFeatureExtractor().to(device)

    with torch.no_grad():
        content_feats = extractor(normalize_for_vgg(content_img))
        style_feats = extractor(normalize_for_vgg(style_img))

    generated = content_img.clone().requires_grad_(True)
    optimizer = Adam([generated], lr=lr)

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        gen_clamped = generated.clamp(0, 1)
        gen_feats = extractor(normalize_for_vgg(gen_clamped))

        c_loss = content_loss(gen_feats[CONTENT_LAYER], content_feats[CONTENT_LAYER])

        s_loss = 0.0
        for layer in STYLE_LAYERS:
            s_loss = s_loss + style_loss(gen_feats[layer], style_feats[layer])

        tv_loss = total_variation_loss(gen_clamped)

        loss = (
            content_weight * c_loss
            + style_weight * s_loss
            + tv_weight * tv_loss
        )

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            generated.clamp_(0, 1)

        if step % log_every == 0 or step == 1 or step == num_steps:
            print(
                f"[Vanilla NST] Step {step:04d}/{num_steps} | "
                f"Content: {c_loss.item():.4f} | "
                f"Style: {s_loss.item():.4f} | "
                f"TV: {tv_loss.item():.6f}"
            )

    return generated.detach().clamp(0, 1)