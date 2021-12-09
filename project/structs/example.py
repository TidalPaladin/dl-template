#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Iterable, TypeVar
from combustion.util.dataclasses import TensorDataclass, BatchMixin

I = TypeVar("I", bound="Example")

@dataclass
class Example(TensorDataclass, BatchMixin):
    ...
