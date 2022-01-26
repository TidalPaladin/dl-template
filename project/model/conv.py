#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

from ..structs import Example, Loss, MultiClassPrediction, Prediction
from .base import BaseModel


class ConvModel(BaseModel[Example, Prediction, Loss]):
    def __init__(self, lr: float = 1e-3, weight_decay: float = 0):
        super().__init__(lr, weight_decay)
        self.num_classes = 10
        self.tail = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.body = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(64, self.num_classes, 1))

    def forward(self, example: Example) -> Prediction:
        img = example.img
        N = img.shape[0]
        x = self.tail(img)
        x = self.body(x)
        x = self.head(x).view(N, -1)

        pred = MultiClassPrediction(logits=x)
        return pred
