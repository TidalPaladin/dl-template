#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict
from torch import Tensor

from combustion.lightning import HydraMixin
from pytorch_lightning.metrics.classification import Accuracy, F1
from pytorch_lightning.metrics import MetricCollection


log = logging.getLogger(__name__)


# define the model
class FakeModel(HydraMixin, pl.LightningModule):
    def __init__(self, in_features: int, out_features: int, num_classes: int = 10, kernel: int = 3):
        super(FakeModel, self).__init__()
        # call self.save_hyperparameters() so that checkpoint loading will work correctly
        self.save_hyperparameters()

        # model layers
        self.l1 = nn.Conv2d(in_features, out_features, kernel)
        self.l2 = nn.AdaptiveAvgPool2d(1)
        self.l3 = nn.Linear(out_features, num_classes)

        # train metrics
        # these can be removed if desired, test metrics are more important
        self.train_metrics = MetricCollection({
            "Accuracy": Accuracy(compute_on_step=True, dist_sync_on_step=False),
            "F1": F1(num_classes=num_classes, compute_on_step=True, dist_sync_on_step=False),
        })

        # test metrics
        # use compute_on_step=False, dist_sync_on_step=True to compute metrics
        # across the entire validation / test set. works in multi-GPU environments
        self.test_metrics = MetricCollection({
            "Accuracy": Accuracy(compute_on_step=False, dist_sync_on_step=True),
            "F1": F1(num_classes=num_classes, compute_on_step=False, dist_sync_on_step=True),
        })


    def forward(self, inputs: Tensor) -> Tensor:
        _ = self.l1(inputs)
        _ = self.l2(_).squeeze()
        _ = self.l3(_)
        return F.relu(_)


    def criterion(self, pred: Tensor, true: Tensor) -> Tensor:
        r"""Criterion computation function. This is included as a separate function in the event
        that loss computation requires additional logic. For trivial cases, loss can be computed
        directly in ``step`` methods. Suggest returning a dict of str -> Tensor for multiple
        losses.

        Args:
            pred (:class:`torch.Tensor`)
                Predicted values

            true (:class:`torch.Tensor`)
                Ground truth values

        Returns:
            Computed loss as a scalar :class:`torch.Tensor`
        """
        return F.cross_entropy(pred, true)

    def compute_metrics(self, pred: Tensor, true: Tensor) -> Dict[str, Tensor]:
        metrics = self.train_metrics if self.training else self.test_metrics
        pred = pred.softmax(dim=-1)
        for name, m in metrics.items():
            m.update(pred, true)
        self.log_dict(metrics, on_step=False, on_epoch=True)

    def _base_step(self, mode: str, batch: Any, batch_nb: int) -> Tensor:
        assert mode in ('train', 'val', 'test'), mode
        img, label = batch

        # forward pass
        pred = self(img)

        # compute loss
        loss = self.criterion(pred, label)

        with torch.no_grad():
            # compute metrics
            self.compute_metrics(pred, label)
            self.log(f"{mode}/loss", loss, on_step=self.training, on_epoch=not self.training)

        # OPTIONAL: set attributes to be logged with combustion callbacks
        #self.last_img = img
        #self.last_pred = pred

        return loss

    def training_step(self, batch, batch_nb):
        self.log("train/lr", self.get_lr(), prog_bar=True)
        return self._base_step('train', batch, batch_nb)

    def validation_step(self, batch, batch_nb):
        return self._base_step('val', batch, batch_nb)

    def test_step(self, batch, batch_nb):
        return self._base_step('test', batch, batch_nb)
