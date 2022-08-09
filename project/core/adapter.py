#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractclassmethod
from typing import Generic, List, Optional, TypeVar, Union, cast

import torch.nn as nn
from registry import Registry
from torch import Tensor


ADAPTER_REGISTRY: Registry = Registry("adapters")


T = TypeVar("T", Tensor, List[Tensor])


class SequentialList(nn.Sequential):
    r"""Variant of :class:`nn.Sequential` that returns the output tensor from each stage
    in the sequence.
    """

    def forward(self, x: Tensor) -> List[Tensor]:
        result: List[Tensor] = []
        for layer in self:
            x = layer(x)
            result.append(x)
        return result

    @classmethod
    def from_sequential(cls, module: nn.Sequential) -> "SequentialList":
        return cls(*[layer for layer in module])


class Backbone(nn.Module, Generic[T]):
    def __init__(self, stem: nn.Module, body: nn.Module, num_features: Optional[int] = None):
        super().__init__()
        self.stem = stem
        self.body = body
        self.num_features = num_features

    def forward(self, x: Tensor) -> T:
        x = self.stem(x)
        x = self.body(x)
        return cast(T, x)


class BackboneAdapter(ABC, Generic[T]):
    r"""A :class:`BackboneAdapter` adapts a model from a provider into a particular form.

    The adapted backbone should have:
        * A stem
        * A body

    """

    @classmethod
    def load_backbone(cls, model_name: str, pretrained: Union[bool, str] = False, *args, **kwargs) -> Backbone[T]:
        model = cls.load_model(model_name, pretrained, *args, **kwargs)
        stem = cls.extract_stem(model)
        body = cls.extract_body(model)
        num_features = cls.extract_num_features(model_name, model)
        return Backbone(stem, body, num_features)

    @abstractclassmethod
    def load_model(cls, model_name: str, pretrained: Union[bool, str] = False, *args, **kwargs) -> nn.Module:
        ...

    @abstractclassmethod
    def extract_stem(cls, model: nn.Module) -> nn.Module:
        ...

    @abstractclassmethod
    def extract_body(cls, model: nn.Module) -> nn.Module:
        ...

    @abstractclassmethod
    def extract_num_features(cls, model_name: str, model: nn.Module) -> int:
        ...
