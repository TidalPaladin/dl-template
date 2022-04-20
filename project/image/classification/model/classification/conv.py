#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from ...structs import Example, Loss, MultiClassPrediction, Prediction
from ..base import BaseModel
from ..decoders import ConvClassifier
from ..encoders import ConvEncoder


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
            optimizer_init,
            lr_scheduler_init,
            lr_scheduler_interval,
            lr_scheduler_monitor,
        )
        self.save_hyperparameters()
        self.encoder = ConvEncoder(width)
        self.classifier = ConvClassifier(num_classes, width)

    def forward(self, example: Example) -> Prediction:
        img = example.img
        x = self.encoder(img)
        x = self.classifier(x)
        pred = MultiClassPrediction(logits=x)
        return pred
