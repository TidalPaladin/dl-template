#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Tuple

from ..structs import Mode


DatasetID = Tuple[Mode, str]


class NamedDataModuleMixin:
    r"""Mixin for LightningDataModules that associate names with each dataset."""

    _lookup: Dict[Mode, List[str]] = {}

    @property
    def name(self) -> str:
        return self.__class__.__name__

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
            if len(seq) == 1:
                return seq[0]
            else:
                return self.name

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
