#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractproperty
from typing import Dict, Iterator, List, Optional, Tuple

from ..structs import Mode


DatasetID = Tuple[Mode, str]


class NamedDataModuleMixin(ABC):
    r"""Mixin for LightningDataModules that associate names with each dataset."""
    _lookup: Dict[Mode, List[str]]

    @abstractproperty
    def name(self) -> str:
        ...

    def register_name(self, mode: Mode, name: str) -> None:
        seq = self._lookup.get(mode, [])
        seq.append(name)
        self._lookup[mode] = seq

    def get_name(self, mode: Mode, dataloader_idx: Optional[int] = None) -> str:
        if mode not in self._lookup:
            raise KeyError(f"No names were defined for mode {mode}")
        seq = self._lookup[mode]

        # single dataloader
        if dataloader_idx is None:
            if len(seq) > 1:
                raise ValueError("dataloader_idx cannot be None if more than one name is registered for a mode")
            else:
                return seq[0]

        # multiple dataloader
        else:
            if not (0 <= dataloader_idx < len(seq)):
                raise IndexError(f"dataloader_idx {dataloader_idx} is out of bounds for names {seq}")
            else:
                return seq[dataloader_idx]

    @property
    def all_names(self) -> Iterator[str]:
        for seq in self._lookup.values():
            for name in seq:
                yield name

    def names_for_mode(self, mode: Mode) -> Iterator[str]:
        if mode not in self._lookup:
            return
        for name in self._lookup[mode]:
            yield name
