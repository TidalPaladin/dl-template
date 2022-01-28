#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

from ..structs import Example, Loss, MultiClassPrediction, Prediction
from .base import BaseModel


class ConvModel(BaseModel[Example, Prediction, Loss]):
    def __init__(self, width: int = 16):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = 10

        W = width
        self.tail = nn.Sequential(
            nn.Conv2d(3, W, 3, stride=2, padding=1),
            nn.BatchNorm2d(W),
            nn.ReLU(),
        )

        self.body = nn.Sequential(
            nn.Conv2d(W, 2 * W, 3, stride=1, padding=1),
            nn.BatchNorm2d(2 * W),
            nn.ReLU(),
            nn.Conv2d(2 * W, 4 * W, 3, stride=2, padding=1),
            nn.BatchNorm2d(4 * W),
            nn.ReLU(),
            nn.Conv2d(4 * W, 4 * W, 3, stride=1, padding=1),
            nn.BatchNorm2d(4 * W),
            nn.ReLU(),
            nn.Conv2d(4 * W, 8 * W, 3, stride=2, padding=1),
            nn.BatchNorm2d(8 * W),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Dropout2d(p=0.2), nn.Conv2d(8 * W, self.num_classes, 1)
        )

    def forward(self, example: Example) -> Prediction:
        img = example.img
        N = img.shape[0]
        x = self.tail(img)
        x = self.body(x)
        x = self.head(x).view(N, -1)

        pred = MultiClassPrediction(logits=x)
        return pred
