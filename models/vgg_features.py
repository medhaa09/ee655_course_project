from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class VGGFeatureExtractor(nn.Module):
    """
    Extracts intermediate feature maps from pretrained VGG19.
    """

    def __init__(self):
        super().__init__()
        weights = VGG19_Weights.DEFAULT
        self.features = vgg19(weights=weights).features.eval()

        for param in self.features.parameters():
            param.requires_grad = False

        # layer mapping for torchvision VGG19 features
        self.layer_name_map = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",   # content layer
            "28": "conv5_1",
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in self.layer_name_map:
                outputs[self.layer_name_map[name]] = x
        return outputs