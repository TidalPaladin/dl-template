#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn as nn

from project.model.decoders import Classifier, ConvClassifier


class TestClassifier:
    @pytest.fixture
    def inputs(self):
        return torch.rand(128, 3, 64)

    @pytest.mark.parametrize("training", [True, False])
    @pytest.mark.parametrize("num_classes", [2, 10, 15])
    @pytest.mark.parametrize("pool", [None, torch.mean, torch.amax])
    def test_forward(self, inputs, training, num_classes, pool):
        inputs.requires_grad = training
        model = Classifier(num_classes, 32, pool=pool)
        if not training:
            model = model.eval()
        out = model(inputs)

        C = num_classes if num_classes > 2 else 1
        assert out.shape == ((128, 3, C) if pool is None else (3, C))
        assert not training or out.requires_grad
        assert not out.isnan().any()


class TestConvClassifier:
    @pytest.fixture
    def inputs(self):
        return torch.rand(3, 64, 4, 4)

    @pytest.mark.parametrize("training", [True, False])
    @pytest.mark.parametrize("num_classes", [2, 10, 15])
    @pytest.mark.parametrize("pool", [nn.AdaptiveAvgPool2d((1, 1)), nn.Identity()])
    def test_forward(self, inputs, training, num_classes, pool):
        inputs.requires_grad = training
        model = ConvClassifier(num_classes, 32, pool=pool)
        if not training:
            model = model.eval()
        out = model(inputs)

        C = num_classes if num_classes > 2 else 1
        assert out.shape == ((3, C, 4, 4) if isinstance(pool, nn.Identity) else (3, C))
        assert not training or out.requires_grad
        assert not out.isnan().any()
