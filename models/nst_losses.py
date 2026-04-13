import torch
import torch.nn.functional as F

from utils.mask_utils import resize_mask


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    features: [B,C,H,W]
    """
    b, c, h, w = features.shape
    x = features.view(b, c, h * w)
    g = torch.bmm(x, x.transpose(1, 2))
    return g / (c * h * w)


def masked_gram_matrix(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    mask: [B,1,H,W] resized to features resolution
    """
    mask = resize_mask(mask, features.shape[-2], features.shape[-1])
    masked_features = features * mask
    return gram_matrix(masked_features)


def content_loss(gen_feat: torch.Tensor, content_feat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(gen_feat, content_feat)


def masked_content_loss(
    gen_feat: torch.Tensor,
    content_feat: torch.Tensor,
    fg_mask: torch.Tensor,
) -> torch.Tensor:
    fg_mask = resize_mask(fg_mask, gen_feat.shape[-2], gen_feat.shape[-1])
    return F.mse_loss(gen_feat * fg_mask, content_feat * fg_mask)


def style_loss(gen_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(gram_matrix(gen_feat), gram_matrix(style_feat))


def masked_style_loss(
    gen_feat: torch.Tensor,
    style_feat: torch.Tensor,
    bg_mask: torch.Tensor,
) -> torch.Tensor:
    gen_gram = masked_gram_matrix(gen_feat, bg_mask)
    style_gram = masked_gram_matrix(style_feat, bg_mask)
    return F.mse_loss(gen_gram, style_gram)


def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """
    Smoothness regularizer.
    """
    loss_h = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    loss_v = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    return loss_h + loss_v