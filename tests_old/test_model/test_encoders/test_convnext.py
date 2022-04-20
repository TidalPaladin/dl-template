#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import torch

from project.model.encoders import ConvNextEncoder


class TestConvNext:
    @pytest.mark.parametrize("training", [True, False])
    def test_forward(self, training):
        B, C, H, W = 3, 3, 256, 256
        inputs = torch.rand(B, C, H, W, requires_grad=training)
        model = ConvNextEncoder.predefined("tiny", pretrained=False)
        if not training:
            model = model.eval()
        out = model(inputs)
        assert len(out) == 4
        assert out[-1].shape == (B, 768, 8, 8)
        assert not training or out[-1].requires_grad
        assert not out[-1].isnan().any()
