#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy

import torch.nn as nn
from torch import Tensor
from typing import Callable, ParamSpec, Dict, List, cast, TypeVar, Generic, Union
from flash.image.classification.backbones import IMAGE_CLASSIFIER_BACKBONES
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE
from flash.core.utilities.imports import requires
from flash.core.utilities.providers import _TORCHVISION
from flash.core.registry import FlashRegistry
from flash.core.utilities.url_error import catch_url_error
from functools import partial
from project.core.adapter import BackboneAdapter, SequentialList

if _TORCHVISION_AVAILABLE:
    import torchvision
    from torchvision.models.convnext import ConvNeXt, convnext_base, convnext_large, convnext_small, convnext_tiny

T = TypeVar("T", Tensor, List[Tensor])

CONVNEXT_MODELS = ["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"]


class ConvNextAdapter(BackboneAdapter[List[Tensor]]):
    r"""A :class:`BackboneAdapter` adapts a model from a provider
    into a particular form.

    The adapted backbone should have:
        * A stem
        * A body

    """

    @requires("torchvision")
    @classmethod
    def load_model(cls, model_name: str, pretrained: Union[bool, str] = False, *args, **kwargs) -> nn.Module:
        module = torchvision.models.convnext
        model = getattr(module, model_name)(pretrained, *args, **kwargs)
        return model

    @classmethod
    def extract_stem(cls, model: nn.Module) -> nn.Module:
        features = cast(List[nn.Module], model.features)
        return features[0]

    @classmethod
    def extract_body(cls, model: nn.Module) -> nn.Module:
        features = cast(List[nn.Module], model.features)
        layers = nn.ModuleList([
            l for i, l in enumerate(features[1:]) if i % 2 == 0
        ])
        downsample = nn.ModuleList([
            l for i, l in enumerate(features[1:]) if i % 2 == 1
        ])
        assert len(layers) == len(downsample) + 1
        blocks = SequentialList([
            nn.Sequential(l, ds) for l, ds in zip(layers, downsample)
        ])
        return SequentialList(*blocks, layers[-1])

    @classmethod
    def extract_num_features(cls, model_name: str, model: nn.Module) -> int:
        assert False


def register_convnext_backbones(register: FlashRegistry):
    for model_name in CONVNEXT_MODELS:
        register(
            fn=catch_url_error(partial(ConvNextAdapter.load_backbone, model_name=model_name)),
            name=model_name,
            namespace="vision",
            package="multiple",
            providers=_TORCHVISION,
            type="convnext",
        )
