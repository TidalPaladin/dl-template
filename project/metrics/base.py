#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torchmetrics as tm
from ..structs import Example, Prediction, Mode
from typing import Optional
from abc import abstractmethod


class DataclassMetricMixin:
    r"""Mixin for :class`tm.Metric` subclasses that implement :func:`update` using
    :class:`Prediction` and :class:`Example` dataclasses as input
    """

    @abstractmethod
    def update(self, example: Example, pred: Prediction) -> None:
        ...
