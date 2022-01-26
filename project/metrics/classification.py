#!/usr/bin/env python
# -*- coding: utf-8 -*-
from torch import Tensor
import torch.nn as nn
import torchmetrics as tm
from ..structs import Example, Prediction, Mode
from typing import Optional, Tuple, Iterable
from abc import abstractmethod
from .base import DataclassMetricMixin
import combustion.lightning.metrics as cm
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt



class Accuracy(tm.Accuracy, DataclassMetricMixin):

    def update(self, example: Example, pred: Prediction) -> None:
        if not example.has_label:
            return
        p = pred.probs
        t = example.label.view(-1)
        assert t is not None
        super().update(p, t)


class UCE(cm.UCE, DataclassMetricMixin):

    def update(self, example: Example, pred: Prediction) -> None:
        if not example.has_label:
            return
        p = pred.logits
        t = example.label.view(-1)
        assert t is not None
        super().update(p, t)


class Entropy(cm.Entropy, DataclassMetricMixin):

    def update(self, example: Example, pred: Prediction) -> None:
        p = pred.logits
        super().update(p)


class ErrorAtUncertainty(cm.ErrorAtUncertainty, DataclassMetricMixin):

    def update(self, example: Example, pred: Prediction) -> None:
        if not example.has_label:
            return
        p = pred.logits
        t = example.label.view(-1)
        assert t is not None
        super().update(p, t)


class ConfusionMatrix(tm.ConfusionMatrix, DataclassMetricMixin):

    def update(self, example: Example, pred: Prediction) -> None:
        if not example.has_label:
            return
        p = pred.logits
        t = example.label.view(-1)
        assert t is not None
        super().update(p, t)

    @staticmethod
    def plot(
        conf_mat: Tensor, 
        display_labels: Optional[Iterable[str]] = None,
        ax: Optional[plt.Axes] = None
    ) -> Optional[plt.Figure]:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        cm = ConfusionMatrixDisplay(conf_mat.cpu().numpy(), display_labels=display_labels)
        cm.plot(ax=ax, colorbar=False) 
        return fig
