#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torchmetrics as tm
from ..structs import Example, Prediction, Mode
from typing import Optional
from abc import abstractmethod

class DataclassMetricMixin:

    def update(self, pred: Prediction, true: Example) -> None:
        ...

class MetricCollection(nn.Module):

    def __init__(self):
        self.lookup = nn.ModuleDict()

    @abstractmethod
    def register(self, mode: Mode, dataset: Optional[str] = None):
        ...

    def update(self, *args, **kwargs) -> None:
        ...

