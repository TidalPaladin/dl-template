#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from enum import Enum
from abc import abstractmethod, abstractproperty, abstractclassmethod
from ..structs import Example, Prediction, State, I, O, L, Loss, MultiClassPrediction, Mode, ResizeMixin, BinaryPrediction, MultiClassPrediction
from ..metrics import StateCollection, MetricStateCollection, QueueStateCollection, PrioritizedItem, DataclassMetricMixin, ErrorAtUncertainty, ConfusionMatrix
from typing import TypeVar, Generic, Any, Type, ClassVar, cast, Optional, List, Set, Iterator, Dict, Tuple, Union, Iterable, TYPE_CHECKING, ForwardRef
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from functools import wraps
from combustion.util import MISSING
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from queue import PriorityQueue
from dataclasses import dataclass, field
from pytorch_lightning.callbacks import Callback
from .base import LoggingTarget, QueuedLoggingCallback, IntervalLoggingCallback, MetricLoggingTarget, MetricLoggingCallback
from .image import ImageTarget, WandBImageTarget
import wandb
from pytorch_lightning.loggers import LightningLoggerBase
from tqdm import tqdm
from PIL import Image
from torchmetrics import MetricCollection
import io

T = TypeVar("T", bound="ImageTarget")


if TYPE_CHECKING:
    from ..model.base import BaseModel
else:
    BaseModel = ForwardRef("BaseModel")


@dataclass
class ErrorAtUncertaintyTarget(MetricLoggingTarget):

    def log(
        self, 
        pl_module: BaseModel, 
        tag: str,
        step: int,
    ) -> None:
        collection_out = self.metric.compute()
        entropy, err, has_items = next(iter(collection_out.values()))
        return self._log(pl_module, tag, entropy, err)

    @rank_zero_only
    def _log(
        self, 
        pl_module: BaseModel, 
        tag: str,
        entropy: Tensor,
        err: Tensor
    ) -> None:
        fig = ErrorAtUncertainty.plot(entropy, err)
        pl_module.wrapped_log({tag: wandb.Image(fig)})


class ErrorAtUncertaintyCallback(MetricLoggingCallback):

    def __init__(
        self, 
        name: str,
        modes: Iterable[Mode],
        log_on_step: bool = False,
        **kwargs
    ):
        kwargs["from_logits"] = True
        kwargs.setdefault("num_bins", 10)
        metric = MetricCollection({name: ErrorAtUncertainty(**kwargs)})
        super().__init__(name, modes, metric, ErrorAtUncertaintyTarget, log_on_step)

@dataclass
class ConfusionMatrixTarget(MetricLoggingTarget):

    def log(
        self, 
        pl_module: BaseModel, 
        tag: str,
        step: int,
    ) -> None:
        collection_out = self.metric.compute()
        mat = next(iter(collection_out.values()))
        return self._log(pl_module, tag, mat)

    @rank_zero_only
    def _log(
        self, 
        pl_module: BaseModel, 
        tag: str,
        mat: Tensor,
    ) -> None:
        fig = ConfusionMatrix.plot(mat)
        pl_module.wrapped_log({tag: wandb.Image(fig)})

class ConfusionMatrixCallback(MetricLoggingCallback):

    def __init__(
        self, 
        name: str,
        modes: Iterable[Mode],
        log_on_step: bool = False,
        **kwargs,
    ):
        metric = MetricCollection({name: ConfusionMatrix(**kwargs)})
        super().__init__(name, modes, metric, ConfusionMatrixTarget, log_on_step)
