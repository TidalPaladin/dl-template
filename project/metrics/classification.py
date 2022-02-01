#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import torchmetrics as tm
from sklearn.metrics import ConfusionMatrixDisplay
from torch import Tensor

import combustion.lightning.metrics as cm

from ..structs import Example, Prediction
from .base import DataclassMetricMixin


class Accuracy(tm.Accuracy, DataclassMetricMixin):
    def update(self, example: Example, pred: Prediction) -> None:
        if example.label is None:
            return
        p = pred.probs
        t = example.label.view(-1)
        assert t is not None
        super().update(p, t)


class UCE(cm.UCE, DataclassMetricMixin):
    def update(self, example: Example, pred: Prediction) -> None:
        if example.label is None:
            return
        p = pred.logits.float()
        t = example.label.view(-1)
        assert t is not None
        super().update(p, t)


class Entropy(cm.Entropy, DataclassMetricMixin):
    def update(self, example: Example, pred: Prediction) -> None:
        p = pred.logits.float()
        super().update(p)


class ErrorAtUncertainty(cm.ErrorAtUncertainty, DataclassMetricMixin):
    def update(self, example: Example, pred: Prediction) -> None:
        if example.label is None:
            return
        p = pred.logits.float()
        t = example.label.view(-1)
        assert t is not None
        super().update(p, t)


class ConfusionMatrix(tm.ConfusionMatrix, DataclassMetricMixin):
    def update(self, example: Example, pred: Prediction) -> None:
        if example.label is None:
            return
        p = pred.logits
        t = example.label.view(-1)
        assert t is not None
        super().update(p, t)

    @staticmethod
    def plot(
        conf_mat: Tensor, display_labels: Optional[Iterable[str]] = None, ax: Optional[plt.Axes] = None
    ) -> Optional[plt.Figure]:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        # TODO ensure the pred/true axes of conf_mat match what ConfusionMatrixDisplay expects
        cm = ConfusionMatrixDisplay(conf_mat.cpu().numpy(), display_labels=display_labels)
        cm.plot(ax=ax, colorbar=False)
        return fig
