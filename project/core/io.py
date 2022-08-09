#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flash.core.data.io.input import DataKeys, Input
from flash.core.data.io.output import Output
from flash.core.data.io.output_transform import OutputTransform
from flash.core.model import OutputKeys
from registry import Registry


INPUT_REGISTRY = Registry("inputs")
OUTPUT_REGISTRY = Registry("outputs")

__all__ = ["Output", "OutputTransform", "OutputKeys", "DataKeys", "Input", "INPUT_REGISTRY", "OUTPUT_REGISTRY"]
