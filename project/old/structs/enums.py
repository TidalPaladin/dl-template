#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, List, Optional, Set, Union


# NOTE: LightingCLI won't accept this as a Callback param annotation
ModeGroup = Iterable[Union["Mode", str]]


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

    @classmethod
    def from_str(cls, val: str) -> "Mode":
        val = val.strip().lower()
        for mode in cls:
            if mode.prefix == val:
                return mode
        raise ValueError(val)

    @classmethod
    def from_group(cls, group: ModeGroup) -> List["Mode"]:
        result: List[Mode] = []
        for item in group:
            if isinstance(item, Mode):
                result.append(item)
            elif isinstance(item, str):
                result.append(cls.from_str(item))
            else:
                raise TypeError(f"Expected Mode or str, found {type(item)}")
        return result


@dataclass(frozen=True)
class State:
    mode: Mode = Mode.INFER
    dataset: Optional[str] = None
    sanity_checking: bool = False

    _seen_datasets: Set[str] = field(default_factory=set, repr=False)

    def __eq__(self, other: "State") -> bool:
        r"""Two states are equal if they have the same ``mode`` and ``dataset``"""
        return self.mode == other.mode and self.dataset == other.dataset

    def __hash__(self) -> int:
        r"""State hash is based on ``mode`` and ``dataset``"""
        return hash(self.mode) + hash(self.dataset)

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
