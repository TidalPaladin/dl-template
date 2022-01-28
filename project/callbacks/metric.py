#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import TYPE_CHECKING, ForwardRef, Iterable, TypeVar, Union

import wandb
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY
from torch import Tensor
from torchmetrics import MetricCollection

from ..metrics import ConfusionMatrix, ErrorAtUncertainty
from ..structs import Mode
from .base import MetricLoggingCallback, MetricLoggingTarget
from .image import ImageTarget


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
        entropy = entropy[has_items]
        err = err[has_items]
        return self._log(pl_module, tag, entropy, err)

    @rank_zero_only
    def _log(self, pl_module: BaseModel, tag: str, entropy: Tensor, err: Tensor) -> None:
        fig = ErrorAtUncertainty.plot(entropy, err)
        pl_module.wrapped_log({tag: wandb.Image(fig)})


@CALLBACK_REGISTRY
class ErrorAtUncertaintyCallback(MetricLoggingCallback):
    def __init__(
        self,
        name: str,
        modes: Iterable[Union[str, Mode]],
        log_on_step: bool = False,
        **kwargs,
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


@CALLBACK_REGISTRY
class ConfusionMatrixCallback(MetricLoggingCallback):
    def __init__(
        self,
        name: str,
        modes: Iterable[Union[str, Mode]],
        num_classes: int,
        log_on_step: bool = False,
        **kwargs,
    ):
        metric = MetricCollection({name: ConfusionMatrix(num_classes, **kwargs)})
        super().__init__(name, modes, metric, ConfusionMatrixTarget, log_on_step)
