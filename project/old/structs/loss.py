#!/usr/bin/env python
# -*- coding: utf-8 -*-


from dataclasses import dataclass
from typing import TypeVar

from torch import Tensor

from combustion.util.dataclasses import BatchMixin, TensorDataclass


L = TypeVar("L", bound="Loss")


@dataclass
class Loss(TensorDataclass, BatchMixin):
    r"""Base container for losses"""
    cls_loss: Tensor

    @property
    def total_loss(self) -> Tensor:
        r"""Returns a total loss value suitable for backprop."""
        return self.cls_loss

    def __len__(self) -> int:
        assert self.is_batched
        return self.cls_loss.shape[0]

    @property
    def is_batched(self) -> bool:
        raise NotImplementedError()

    @classmethod
    def from_unbatched(cls):
        raise NotImplementedError()
