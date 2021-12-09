#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum

class Mode(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2
    INFER = 3

    @property
    def prefix(self) -> str:
        return str(self.name.lower())
