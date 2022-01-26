#!/usr/bin/env python
# -*- coding: utf-8 -*-


from abc import abstractproperty
from dataclasses import dataclass
from typing import TypeVar

from torch import Tensor

from combustion.util.dataclasses import BatchMixin, TensorDataclass


L = TypeVar("L", bound="Loss")


@dataclass
class Loss(TensorDataclass, BatchMixin):
    r"""Base container for losses"""
    cls_loss: Tensor

    @abstractproperty
    def total_loss(self) -> Tensor:
        r"""Returns a total loss value suitable for backprop."""
        return self.cls_loss
