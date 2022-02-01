#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import Tensor
from project.model.vit import MaskIndices


def test_mask_indices():
    N, H, W = 4, 32, 32
    idx = MaskIndices.create(0.1, 0.25, N)
    x = torch.rand(N, 3, H, W)
    foo = idx.select(x)
    assert isinstance(foo, Tensor)

def test_slice():
    N, H, W = 4, 32, 32
    idx = MaskIndices.create(0.1, 0.25, N)
    sliced = idx[0]

    x = torch.rand(3, H, W)
    foo = sliced.select(x)
    assert isinstance(foo, Tensor)
