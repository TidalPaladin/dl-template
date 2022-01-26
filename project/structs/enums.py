#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum
from dataclasses import dataclass, field, replace
from typing import Optional, Set

class Mode(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2
    INFER = 3

    @property
    def prefix(self) -> str:
        return str(self.name.lower())

    def __repr__(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class State:
    mode: Mode = Mode.INFER
    dataset: Optional[str] = None
    sanity_checking: bool = False

    _seen_datasets: Set[str] = field(default_factory=set, repr=False, hash=False)

    def update(self, mode: Mode, dataset: Optional[str]) -> "State":
        return self.set_mode(mode).set_dataset(dataset)

    def set_mode(self, mode: Mode) -> "State":
        return replace(self, mode=mode)

    def set_dataset(self, name: Optional[str]) -> "State":
        if name is None:
            return replace(self, dataset=None)
        seen_datasets = self._seen_datasets.union({name})
        return replace(self, dataset=name, _seen_datasets=seen_datasets)

    def set_sanity_checking(self, value: bool) -> "State":
        return replace(self, sanity_checking=value)

    @property
    def prefix(self) -> str:
        # TODO: are we sure we want to hide train dataset name?
        if self.mode == Mode.TRAIN or self.dataset is None:
            return f"{self.mode.prefix}/"
        else:
            return f"{self.mode.prefix}/{self.dataset}/"

    def with_postfix(self, postfix: str) -> str:
        return f"{self.prefix}{postfix}"
