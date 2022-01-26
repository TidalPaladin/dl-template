#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from enum import Enum
from abc import abstractmethod, abstractproperty
from ...structs import Example, Prediction, Mode, I, O, L, Loss
from typing import TypeVar, Generic, Any, Type, ClassVar, cast, Optional
import pytorch_lightning as pl
from functools import wraps
from combustion.util import MISSING
from .other import ModeMixin
import torchmetrics as tm




class MetricMixin(Generic[I, O], ModeMixin):
    r"""Base class for all models."""
    mode: Mode = Mode.INFER
    metric_lookup = nn.ModuleDict()

    @property
    def current_metrics(self) -> Optional[tm.MetricCollection]:
        r"""Gets a collection of metrics applicable to the current mode of operation"""
        if self.current_prefix in self.metric_lookup.keys():
            return cast(MetricCollection, self.metric_lookup[self.current_prefix()])
        return None

    @abstractmethod
    def compute_metrics(self, example: I, pred: O) -> L:
        ...

    def step(self, example: I, batch_idx: int, *args, **kwargs):
        pred = self(example)
        loss = self.compute_loss(example, pred)
        return loss.total_loss

