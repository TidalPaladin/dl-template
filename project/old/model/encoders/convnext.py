#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy

import torch.nn as nn
from torch import Tensor
from torchvision.models.convnext import ConvNeXt, convnext_base, convnext_large, convnext_small, convnext_tiny
from typing import Callable, ParamSpec, Dict, List, cast


MODEL_LOOKUP: Dict[str, Callable] = {
    "tiny": convnext_tiny,
    "small": convnext_small,
    "base": convnext_base,
    "large": convnext_large,
}


class ConvNextEncoder(nn.Module):
    r"""Basic convolutional backbone"""

    def __init__(self, proto: ConvNeXt):
        super().__init__()
        features = cast(List[nn.Module], proto.features)

        # extract stem, blocks, and downampling from torchvision model
        self.stem = features[0]
        self.layers = nn.ModuleList([
            l for i, l in enumerate(features[1:]) if i % 2 == 0
        ])
        self.downsample = nn.ModuleList([
            l for i, l in enumerate(features[1:]) if i % 2 == 1
        ])

    def forward(self, img: Tensor) -> List[Tensor]:
        features: List[Tensor] = []
        x = self.stem(img)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            features.append(x)
            if i < len(self.downsample):
                x = self.downsample[i](x)
        return features

    @classmethod
    def create(cls, *args, **kwargs) -> "ConvNextEncoder":
        model = ConvNeXt(*args, **kwargs)
        return cls(model)

    @classmethod
    def predefined(cls, name: str, *args, **kwargs) -> "ConvNextEncoder":
        if name not in MODEL_LOOKUP:
            raise ValueError(f"Unknown model {name}")
        model = MODEL_LOOKUP[name](*args, **kwargs)
        return cls(model)
