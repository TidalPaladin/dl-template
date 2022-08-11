#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import Callable, Optional, Union, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from timm.models.vision_transformer import VisionTransformer
from einops import rearrange

from ...core import HEAD_REGISTRY


@HEAD_REGISTRY(name="vit-mae")
class ViTDecoder(VisionTransformer):

    def __init__(
        self, 
        num_features: int,
        patch_size: Tuple[int, int],
        embed_dim: int = 512, 
        num_heads: int = 8,
        depth: int = 5, 
        out_channels: int = 3, 
        **kwargs
    ):
        super().__init__(embed_dim=embed_dim, depth=depth, num_heads=num_heads, **kwargs)
        self.patch_size = patch_size
        self.in_proj = nn.Linear(num_features, embed_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * .02)
        self.out_channels = out_channels
        Hp, Wp = self.patch_size
        out_dim = out_channels * Hp * Wp
        self.head = nn.Linear(embed_dim, out_dim)

    def forward(self, features: Tensor, mask: Tensor, img_size: Tuple[int, int]) -> Tensor:
        Hp, Wp = self.patch_size
        H = img_size[0] // Hp
        W = img_size[1] // Wp
        features = self.in_proj(features)
        features = self.create_masked_input(features, mask)
        x = self.head(features)
        x = rearrange(
            x,
            "n (h w) (hp wp c) -> n c (h hp) (w wp)",
            h=H,
            w=W,
            hp=Hp,
            wp=Wp,
            c=self.out_channels,
        )
        return x

    def create_masked_input(self, features: Tensor, mask: Tensor) -> Tensor:
        N, Lf, D = features.shape
        with torch.no_grad():
            mask = mask.view(1, -1, 1).expand(N, -1, D)
        result = torch.empty_like(mask, dtype=features.dtype)
        result[mask] = self.mask_token.expand_as(result).type_as(features)[mask]
        result[~mask] = features.view(-1)
        return result

