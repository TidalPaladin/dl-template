#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flash.core.registry import FlashRegistry  # noqa: F401
from flash.image.classification.backbones import IMAGE_CLASSIFIER_BACKBONES
from .convnext import register_convnext_backbones

register_convnext_backbones(IMAGE_CLASSIFIER_BACKBONES)
