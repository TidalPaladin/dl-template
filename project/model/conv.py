#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from ..structs import Example, Loss, MultiClassPrediction, Prediction
from .base import BaseModel


class ConvModel(BaseModel[Example, Prediction, Loss]):
    r"""Basic convolutional model"""
    example_input_array = Example(img=torch.rand(1, 3, 32, 32))

    def __init__(
        self,
        width: int = 16,
        num_classes: int = 10,
        optimizer_init: dict = {},
        lr_scheduler_init: dict = {},
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_monitor: str = "train/total_loss_epoch",
    ):
        super().__init__(
            num_classes,
            optimizer_init,
            lr_scheduler_init,
            lr_scheduler_interval,
            lr_scheduler_monitor,
        )
        self.save_hyperparameters()

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
