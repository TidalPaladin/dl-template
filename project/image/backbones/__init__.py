#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .basic_conv import ConvEncoder
from .vit import register_vit_backbones
from ...core import BACKBONE_REGISTRY

register_vit_backbones(BACKBONE_REGISTRY)

__all__ = ["ConvEncoder"]
