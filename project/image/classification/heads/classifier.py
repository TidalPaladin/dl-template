#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from ....core import HEAD_REGISTRY


@HEAD_REGISTRY(name="seq-linear")
class Classifier(nn.Module):
    r"""Simple classification head for sequential inputs. The head consists of an optional pooling layer, multiple
    neck blocks, and a final classification linear layer.

    Neck Composition:
        * ``nn.Dropout``
        * ``nn.Linear``
        * ``nn.LayerNorm``
        * ``nn.Mish`` (or selected activation function)

    Args:
        num_classes:
            Number of output classes. If ``num_classes <= 2``, assume binary classification

        width:
            Number of channels in the neck

        depth:
            Number of neck layers

        dropout:
            Dropout rate for neck layers

        pool:
            A pooling function to use, or ``None`` for no pooling

        prior:
            Initialization prior for bias of the final classification layer

        act:
            Activation to use on neck layers

    Shapes:
        * ``x`` - :math:`(L, N, D)`
        * Output - :math:`(L, N, C)` if pooling is disabled, otherwise :math:`(N, C)`
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        depth: int = 1,
        dropout: float = 0.0,
        pool: Optional[Callable] = torch.mean,
        prior: Optional[Union[Tensor, float]] = None,
        act: nn.Module = nn.Mish(),
    ):
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, found {num_classes}")
        self.num_classes = num_classes
        self.pool = pool
        self.neck = self.create_neck(num_features, depth, dropout, act)
        self.head = nn.Linear(
            num_features,
            self.num_classes if self.num_classes > 2 else 1,
        )
        self.reset_parameters(prior)

    def create_neck(self, width: int, depth: int, dropout: float, act: nn.Module) -> nn.Module:
        neck = nn.Sequential()
        for i in range(depth):
            neck.add_module(f"dropout_{i}", nn.Dropout(dropout))
            neck.add_module(f"linear_{i}", nn.LazyLinear(width))
            neck.add_module(f"norm_{i}", nn.LayerNorm(width))
            neck.add_module(f"act_{i}", deepcopy(act))
        return neck

    @property
    def is_multiclass(self) -> bool:
        return self.num_classes > 2

    def reset_parameters(self, prior: Optional[Union[Tensor, float]] = None):
        torch.nn.init.normal_(self.head.weight, std=0.01)
        if prior is None:
            prior = torch.as_tensor(1 / self.num_classes)
        prior = torch.as_tensor(prior)
        bias_value = prior.logit()
        if bias_value.numel() == 1:
            bias_param = self.head.bias
            assert isinstance(bias_param, Tensor)
            torch.nn.init.constant_(bias_param, float(bias_value.item()))
        else:
            self.head.bias = nn.Parameter(bias_value)

    def forward(self, x: Tensor) -> Tensor:
        L, N, D = x.shape
        if self.pool is not None:
            x = self.pool(x, dim=0)
        x = self.neck(x)
        x = self.head(x)
        assert x.shape[-1] == self.num_classes if self.num_classes > 2 else 1
        return x


@HEAD_REGISTRY(name="conv-linear")
class ConvClassifier(Classifier):
    r"""Simple classification head for 2D inputs. The head consists of an optional pooling layer, multiple
    neck blocks, and a final classification pointwise convolution layer.

    Neck Composition:
        * ``nn.Dropout2d``
        * ``nn.Conv2d``
        * ``nn.BatchNorm2d``
        * ``nn.Mish`` (or selected activation function)

    Args:
        num_classes:
            Number of output classes. If ``num_classes <= 2``, assume binary classification

        width:
            Number of channels in the neck

        depth:
            Number of neck layers

        dropout:
            Dropout rate for neck layers

        pool:
            A pooling layer to use. Use :class:`nn.Identity` for no pooling

        prior:
            Initialization prior for bias of the final classification layer

        act:
            Activation to use on neck layers

    Shapes:
        * ``x`` - :math:`(N, D, H, W)`
        * Output - :math:`(N, C, H, W)` if pooling is disabled, otherwise :math:`(N, C)`
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        depth: int = 1,
        dropout: float = 0.0,
        pool: nn.Module = nn.AdaptiveAvgPool2d((1, 1)),
        prior: Optional[Union[Tensor, float]] = None,
        act: nn.Module = nn.Mish(),
    ):
        super().__init__(num_classes, num_features, depth, dropout, pool, prior, act)
        self.pool = pool
        self.head = nn.Conv2d(
            num_features,
            self.num_classes if self.num_classes > 2 else 1,
            kernel_size=1,
        )
        self.reset_parameters(prior)

    def create_neck(self, width: int, depth: int, dropout: float, act: nn.Module) -> nn.Module:
        kernel_size = 1
        neck = nn.Sequential()
        for i in range(depth):
            neck.add_module(f"dropout_{i}", nn.Dropout2d(dropout))
            neck.add_module(f"conv_{i}", nn.LazyConv2d(width, kernel_size))
            neck.add_module(f"norm_{i}", nn.BatchNorm2d(width))
            neck.add_module(f"act_{i}", deepcopy(act))
        return neck

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        x = self.pool(x)
        x = self.neck(x)
        x = self.head(x)
        if not isinstance(self.pool, nn.Identity):
            x = x.view(N, -1)
        assert x.shape[1] == self.num_classes if self.num_classes > 2 else 1
        return x


@HEAD_REGISTRY(name="linear")
class Head(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Linear(
            num_features,
            num_classes if num_classes > 2 else 1,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.head(x)
        assert x.shape[1] == self.num_classes if self.num_classes > 2 else 1
        return x.squeeze(-1)
