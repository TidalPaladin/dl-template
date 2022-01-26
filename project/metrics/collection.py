#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Dict, Generic, List, Optional, Sequence, Set, Tuple, TypeVar, Union, cast

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from torchmetrics import MetricCollection

from ..structs import Mode, State


T = TypeVar("T", bound="StateCollection")
P = TypeVar("P")
M = TypeVar("M", bound="nn.Module")
U = TypeVar("U")


class StateCollection(ABC, Generic[U]):
    r"""Container for storing objects that are associated with a given :class:`State`."""

    @abstractmethod
    def register(self, state: State):
        r"""Register a :class:`MetricCollection` for a given :class:`State`."""
        ...

    @abstractmethod
    def reset(
        self: T,
        specific_states: Sequence[State] = [],
        specific_modes: Sequence[Mode] = [],
    ) -> T:
        r"""Reset contained collections.

        Args:
            specific_states:
                If provided, only reset the specified states

            specific_modes:
                If provided, only reset states with the specified modes

        Returns:
            Reference to reset self
        """
        return self

    @abstractmethod
    def set_state(self, state: State, val: U) -> None:
        r"""Associates a collection ``val`` with state ``State``."""
        ...

    @abstractmethod
    def get_state(self, state: State) -> U:
        r"""Returns the collection associated with state ``State``."""
        ...

    @abstractmethod
    def remove_state(self, state: State) -> None:
        r"""Removes state ``state`` if present."""
        ...

    @abstractproperty
    def states(self) -> Set[State]:
        r"""Returns the set of registered ``State`` keys."""
        ...

    def as_dict(self) -> Dict[State, U]:
        r"""Returns this collection as a simple State -> U dictionary"""
        return {state: self.get_state(state) for state in self.states}

    def clear(self) -> None:
        r"""Clear all states from the collection"""
        for state in self.states:
            self.remove_state(state)

    def __add__(self: T, other: T) -> T:
        r"""Join two StateColletions"""
        output = deepcopy(self)
        for state, val in other.as_dict().items():
            output.set_state(state, val)
        return output


class ModuleStateCollection(nn.ModuleDict, StateCollection[M]):
    r"""Container for storing :class:`nn.Module` instances that are associated with a given
    :class:`State`. Inherits from :class:`nn.ModuleDict` to support stateful attachment of
    contained :class:`nn.Module` instances.
    """

    # NOTE: nn.ModuleDict strictly requires keys to be str and values to be nn.Module
    #   * self._lookup maintains a State -> str mapping, where str is state.prefix
    #   * Target nn.Module is inserted into self using the str key
    #   * State lookups find the nn.Module by State -> str -> nn.Module
    _lookup: Dict[State, str]

    def __init__(self):
        super().__init__()
        self._lookup = {}

    def _get_key(self, state: State) -> str:
        r"""Gets a string key for state ``State``"""
        return state.prefix

    def set_state(self, state: State, val: M) -> None:
        r"""Associates a collection ``val`` with state ``State``."""
        key = self._get_key(state)
        self._lookup[state] = key
        self[key] = val

    def get_state(self, state: State) -> M:
        r"""Returns the collection associated with state ``State``."""
        if state not in self.states:
            raise KeyError(str(state))
        key = self._get_key(state)
        assert key in self.keys()
        return cast(M, self[key])

    def remove_state(self, state: State) -> None:
        r"""Removes state ``state`` if present."""
        if state not in self.states:
            return
        key = self._get_key(state)
        del self[key]
        del self._lookup[state]

    @property
    def states(self) -> Set[State]:
        r"""Returns the set of registered ``State`` keys."""
        return set(self._lookup.keys())


def join_collections(col1: MetricCollection, col2: MetricCollection) -> MetricCollection:
    full_dict = {name: metric for col in (col1, col2) for name, metric in col.items()}
    full_dict = cast(Dict[str, tm.Metric], full_dict)
    return MetricCollection(full_dict)


class MetricStateCollection(ModuleStateCollection[MetricCollection]):
    r"""Container for storing multiple :class:`MetricCollections`, with each collection being
    associated with a given :class:`State` (mode, dataset pair).

    Args:
        collection:
            The base :class:`MetricCollection` to attach when registering a state. If not provided,
            please use :func:`set_state` to assign a collection
    """

    def __init__(self, collection: Optional[MetricCollection] = None):
        super().__init__()
        self._collection = collection

    def register(self, state: State, device: Union[str, torch.device] = "cpu"):
        r"""Register a :class:`MetricCollection` for a given :class:`State`."""
        if state in self.states:
            return
        elif self._collection is None:
            raise ValueError(
                "Value of `collection` in init cannot be `None` to use `register`. "
                "Either supply a `MetricCollection` in init, or manually register collections "
                "with `set_state`"
            )
        device = torch.device(device)
        collection = self._collection.clone(prefix=state.prefix).to(device)
        self.set_state(state, collection)

    def update(self, state: State, *args, **kwargs) -> MetricCollection:
        collection = self.get_state(state)
        collection.update(*args, **kwargs)
        return collection

    @torch.no_grad()
    def log(
        self,
        state: State,
        pl_module: pl.LightningModule,
        on_step: bool = False,
        on_epoch: bool = True,
    ) -> None:
        if state not in self.states:
            return

        collection = self.get_state(state)
        attr = "state_metrics"
        prefix = collection.prefix

        for name, metric in collection.items():
            metric = cast(tm.Metric, metric)
            metric_attribute = f"{attr}.{prefix}.{name}"
            pl_module.log(
                name,
                metric,
                on_step=on_step,
                on_epoch=on_epoch,
                add_dataloader_idx=False,  # type: ignore
                rank_zero_only=True,  # type: ignore
                metric_attribute=metric_attribute,  # type: ignore
            )

    def reset(
        self: T,
        specific_states: Sequence[State] = [],
        specific_modes: Sequence[Mode] = [],
    ) -> T:
        for k, v in self.as_dict().items():
            if specific_states and k not in specific_states:
                continue
            elif specific_modes and k.mode not in specific_modes:
                continue
            v.reset()
        return self

    def __add__(self: T, other: T) -> T:
        output = deepcopy(self)
        for state, val in other.as_dict().items():
            # add unseen states to output
            if state not in self.states:
                output.set_state(state, val)

            # join MetricCollection seen in both containers
            else:
                collection = self.get_state(state)
                other_collection = other.get_state(state)
                joined = join_collections(collection, other_collection)
                output.set_state(state, joined)
        return output

    def summarize(self) -> str:
        lines: List[Tuple[str, str]] = []
        maxlen = 0
        for state in self.states:
            collection = self.get_state(state)
            for name, metric in collection.items():
                lines.append((name, str(metric)))
                maxlen = max(maxlen, len(name))
        fmt = "{0:<" + str(maxlen) + "} -> {1}\n"
        s = ""
        for name, metric in lines:
            s += fmt.format(name, metric)
        return s


@dataclass(order=True)
class PrioritizedItem(Generic[P]):
    priority: Union[int, float]
    item: P = field(compare=False)


class QueueStateCollection(StateCollection[PriorityQueue[PrioritizedItem[P]]]):
    r"""Collection that associates each State with a PriorityQueue."""
    QueueType = PriorityQueue[PrioritizedItem]
    _lookup: Dict[State, QueueType]

    def __init__(self):
        super().__init__()
        self._lookup = {}

    def register(self, state: State, maxsize: int = 0):
        if state in self.states:
            return
        queue = PriorityQueue(maxsize=maxsize)
        self.set_state(state, queue)

    def set_state(self, state: State, val: QueueType) -> None:
        self._lookup[state] = val

    def get_state(self, state: State) -> QueueType:
        r"""Returns the collection associated with state ``State``."""
        if state not in self.states:
            raise KeyError(state)
        return self._lookup[state]

    def remove_state(self, state: State) -> None:
        r"""Removes state ``state`` if present."""
        if state in self.states:
            del self._lookup[state]

    @property
    def states(self) -> Set[State]:
        return set(self._lookup.keys())

    def enqueue(self, state: State, priority: Union[int, float], value: P, *args, **kwargs) -> None:
        item = PrioritizedItem(priority, value)
        queue = self.get_state(state)
        queue.put(item, *args, **kwargs)

    def dequeue(self, state: State, *args, **kwargs) -> PrioritizedItem[P]:
        queue = self.get_state(state)
        return queue.get(*args, **kwargs)

    def empty(self, state: State) -> bool:
        queue = self.get_state(state)
        return queue.empty()

    def qsize(self, state: State) -> int:
        queue = self.get_state(state)
        return queue.qsize()

    def __len__(self) -> int:
        r"""Gets the total number of currently queued items across all states"""
        return sum(self.qsize(state) for state in self.states)

    def reset(
        self: T,
        specific_states: Sequence[State] = [],
        specific_modes: Sequence[Mode] = [],
    ) -> T:
        for k, v in self.as_dict().items():
            if specific_states and k not in specific_states:
                continue
            elif specific_modes and k.mode not in specific_modes:
                continue
            while not v.empty():
                v.get_nowait()
            assert v.empty()
        return self

    def __add__(self: T, other: T) -> T:
        # NOTE: this will modify self inplace - PriorityQueue can't be deepcopied
        output = self
        for state, val in other.as_dict().items():
            output.set_state(state, val)
        return output
