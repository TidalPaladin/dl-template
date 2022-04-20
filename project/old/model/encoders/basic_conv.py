#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy

import torch.nn as nn
from torch import Tensor


class ConvEncoder(nn.Module):
    r"""Basic convolutional backbone"""

    def __init__(
        self,
        width: int = 16,
        act: nn.Module = nn.Mish(),
    ):
        super().__init__()

        W = width
        self.tail = nn.Sequential(
            nn.Conv2d(3, W, 3, stride=2, padding=1),
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

    def forward(self, img: Tensor) -> Tensor:
        img.shape[0]
        x = self.tail(img)
        x = self.body(x)
        return x
