#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import torch

from project.model.encoders import ConvEncoder


class TestBasicConv:
    @pytest.mark.parametrize("training", [True, False])
    def test_forward(self, training):
        B, C, H, W = 3, 3, 32, 32
        inputs = torch.rand(B, C, H, W, requires_grad=training)
        model = ConvEncoder(width=8)
        if not training:
            model = model.eval()
        out = model(inputs)
        assert out.shape == (B, 64, 4, 4)
        assert not training or out.requires_grad
        assert not out.isnan().any()
