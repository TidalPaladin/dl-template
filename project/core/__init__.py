#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .adapter import ADAPTER_REGISTRY
from .io import INPUT_REGISTRY, OUTPUT_REGISTRY, Input, Output, OutputTransform
from .task import (
    BACKBONE_REGISTRY,
    HEAD_REGISTRY,
    LOSS_FN_REGISTRY,
    LR_SCHEDULER_REGISTRY,
    MODEL_REGISTRY,
    OPTIMIZER_REGISTRY,
)


__all__ = [
    "INPUT_REGISTRY",
    "OUTPUT_REGISTRY",
    "Input",
    "Output",
    "OutputTransform",
    "OPTIMIZER_REGISTRY",
    "LOSS_FN_REGISTRY",
    "LR_SCHEDULER_REGISTRY",
    "MODEL_REGISTRY",
    "BACKBONE_REGISTRY",
    "HEAD_REGISTRY",
    "AdapterTask",
    "ADAPTER_REGISTRY",
]
