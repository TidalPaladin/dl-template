#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from enum import Enum
from abc import abstractmethod, abstractproperty
from ...structs import Example, Prediction, Mode, I, O, L, Loss, State
from typing import TypeVar, Generic, Any, Type, ClassVar, cast, Hashable, Dict, Sequence, Optional, Set, Tuple, Iterator
import pytorch_lightning as pl
from functools import wraps
from combustion.util import MISSING
from torchmetrics import MetricCollection


T = TypeVar("T", bound="StateCollection")


class StateCollection:
    r"""Container for storing multiple :class:`MetricCollections`, with each collection being
    associated with a given :class:`State` (mode, dataset pair).
    """
    _state: Optional[State] = None
    _lookup = nn.ModuleDict()

    @abstractmethod
    def register(self, state: State):
        r"""Register a :class:`MetricCollection` for a given :class:`State`."""
        ...

    @property
    def state(self) -> State:
        if self._state is None:
            raise AttributeError("State has not been set")
        return self._state

    @state.setter
    def state(self, state: State) -> None:
        if not isinstance(state, State):
            raise TypeError(f"Expected State for `state`, found {type(state)}")
        self._state = state

    @property
    def lookup(self) -> Dict[State, MetricCollection]:
        # NOTE: we need _lookup to be a nn.ModuleDict so that everything gets copied to
        # the correct device when calling .cuda(), etc. This property provides type 
        # casting
        return cast(Dict[State, MetricCollection], self._lookup)

    @property
    def current_metrics(self) -> MetricCollection:
        r"""Gets a collection of metrics applicable to the current mode of operation"""
        return self.lookup[self.state]

    def items(self) -> Iterator[Tuple[State, MetricCollection]]:
        for item in self.lookup.items():
            yield item

    @property
    def registered_states(self) -> Set[State]:
        return set(self.lookup.keys())

    def to(self, device: torch.device):
        self._lookup = self._lookup.to(device)
        for k, v in self._lookup.named_children():
            assert v.device == device, f"Child {k} was not on device {device}"

    def reset(self: T, specific_states: Sequence[State] = []) -> T:
        for k, v in self.lookup.items():
            if specific_states and k not in specific_states:
                continue
            v.reset()
        return self
