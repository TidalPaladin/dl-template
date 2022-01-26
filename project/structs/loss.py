#!/usr/bin/env python
# -*- coding: utf-8 -*-


from abc import abstractproperty
from torch import Tensor 
from dataclasses import dataclass
from combustion.util.dataclasses import TensorDataclass, BatchMixin

from typing import Iterable, TypeVar

L = TypeVar("L", bound="Loss")

@dataclass
class Loss(TensorDataclass, BatchMixin):
    r"""Base container for losses"""
    cls_loss: Tensor

    @abstractproperty
    def total_loss(self) -> Tensor:
        r"""Returns a total loss value suitable for backprop."""
        return self.cls_loss
