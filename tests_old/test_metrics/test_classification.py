#!/usr/bin/env python
# -*- coding: utf-8 -*-


from typing import Type

import matplotlib.pyplot as plt
import pytest
import torch
from torch import Tensor

from project.metrics import Accuracy, ConfusionMatrix
from project.structs import BinaryPrediction, Example, MultiClassPrediction, Prediction


class BaseMetricTest:
    B = 10
    N = 5

    @pytest.fixture
    def metric(self):
        raise NotImplementedError()

    @pytest.fixture(params=[BinaryPrediction, MultiClassPrediction])
    def pred(self, request):
        B, N = self.B, self.N
        cls: Type[Prediction] = request.param
        torch.random.manual_seed(42)

        if issubclass(cls, BinaryPrediction):
            logits = torch.rand(B, N)
            torch.randint(0, 1, (B, 1))
            pred = cls(logits)

        elif issubclass(cls, MultiClassPrediction):
            logits = torch.rand(B, N)
            torch.randint(0, N, (B, 1))
            pred = cls(logits)

        else:
            raise NotImplementedError(cls)

        return pred

    @pytest.fixture(params=[Example])
    def true(self, request):
        B, N = self.B, self.N
        cls: Type[Example] = request.param
        cls = request.param
        labels = torch.randint(0, N, (B, 1))
        img = torch.rand(B, 1, 32, 32)
        true = cls(img, labels)
        return true

    def test_metric(self, metric, pred, true):
        metric.update(true, pred)
        x = metric.compute()
        assert isinstance(x, Tensor)
        return x


class TestAccuracy(BaseMetricTest):
    @pytest.fixture
    def metric(self):
        return Accuracy(num_classes=self.N)

    def test_metric(self, metric, pred, true):
        x = super().test_metric(metric, pred, true)
        assert 0 <= float(x.item()) <= 1.0


class TestConfusionMatrix(BaseMetricTest):
    @pytest.fixture
    def metric(self):
        return ConfusionMatrix(num_classes=self.N)

    def test_metric(self, metric, pred, true):
        x = super().test_metric(metric, pred, true)
        assert not x.is_floating_point()
        assert (x >= 0).all()
        assert x.shape == (self.N, self.N)

    def test_plot(self, metric, pred, true):
        metric.update(true, pred)
        mat = metric.compute()
        fig = ConfusionMatrix.plot(mat)
        assert isinstance(fig, plt.Figure)
