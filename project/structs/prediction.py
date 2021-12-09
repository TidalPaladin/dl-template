#!/usr/bin/env python
# -*- coding: utf-8 -*-


from dataclasses import dataclass
from typing import Iterable, TypeVar
from combustion.util.dataclasses import TensorDataclass, BatchMixin

O = TypeVar("O", bound="Prediction")

@dataclass
class Prediction(TensorDataclass, BatchMixin):
    ...
