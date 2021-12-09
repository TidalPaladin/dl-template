#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum
from abc import abstractmethod, abstractproperty
from ..structs import Example, Prediction, Mode
from typing import TypeVar, Generic
import pytorch_lightning as pl

I = TypeVar("I", bound=Example)
O = TypeVar("O", bound=Prediction)



class BaseModel(pl.LightningModule, Generic[I, O]):
    mode: Mode = Mode.INFER

    def __init__(self, lr: float = 1e-3, weight_decay: float = 0):
        super().__init__()

    def forward(self, example: I) -> O:
        ...
