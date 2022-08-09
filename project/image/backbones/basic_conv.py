#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy

import torch.nn as nn
from torch import Tensor

from ...core import BACKBONE_REGISTRY


@BACKBONE_REGISTRY(name="basic-conv")
class ConvEncoder(nn.Module):
    r"""Basic convolutional backbone"""

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 16,
        act: nn.Module = nn.Mish(),
    ):
        super().__init__()

        W = width
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, W, 3, stride=2, padding=1),
            nn.BatchNorm2d(W),
            deepcopy(act),
        )

        self.body = nn.Sequential(
            nn.Conv2d(W, 2 * W, 3, stride=1, padding=1),
            nn.BatchNorm2d(2 * W),
            deepcopy(act),
            nn.Conv2d(2 * W, 4 * W, 3, stride=2, padding=1),
            nn.BatchNorm2d(4 * W),
            deepcopy(act),
            nn.Conv2d(4 * W, 4 * W, 3, stride=1, padding=1),
            nn.BatchNorm2d(4 * W),
            deepcopy(act),
            nn.Conv2d(4 * W, 8 * W, 3, stride=2, padding=1),
            nn.BatchNorm2d(8 * W),
            deepcopy(act),
        )
        self.num_features = 8 * W

    def forward(self, img: Tensor) -> Tensor:
        x = self.stem(img)
        x = self.body(x)
        return x
