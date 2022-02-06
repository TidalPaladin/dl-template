#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, ForwardRef, Generic, Iterable, Union

import pytorch_lightning as pl
import torch
import torchmetrics as tm
import wandb
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY
from torchmetrics import MetricCollection

from ..metrics import ConfusionMatrix, ErrorAtUncertainty, MetricStateCollection
from ..structs import I, Mode, O, State
from .base import ALL_MODES, LoggingCallback, ModeGroup


if TYPE_CHECKING:
    from ..model.base import BaseModel
else:
    BaseModel = ForwardRef("BaseModel")


class MetricLoggingCallback(LoggingCallback, ABC, Generic[I, O]):
    r"""Callback for logging"""

    def __init__(
        self,
        name: str,
        collection: tm.MetricCollection,
        modes: ModeGroup = ALL_MODES,
        log_on_step: bool = False,
    ):
        super().__init__(name, modes)
        self.state_metrics = MetricStateCollection(collection)
        self.log_on_step = log_on_step

    @abstractmethod
    def log_target(
        self,
        target: Dict[str, Any],
        pl_module: BaseModel,
        tag: str,
        step: int,
    ):
        ...

    def __len__(self) -> int:
        # it is difficult to track metrics with pending log calls, so just return 0 here
        return 0

    def reset(self, specific_states: Iterable[State] = [], specific_modes: Iterable[Mode] = []):
        self.state_metrics.reset(
            specific_states=list(specific_states),
            specific_modes=list(specific_modes),
        )

    def register(self, state: State, pl_module: BaseModel, *args, **kwargs) -> None:
        if state not in self.state_metrics.states:
            self.state_metrics.register(state, device=torch.device(pl_module.device))

    def _on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: BaseModel,
        outputs: O,
        batch: I,
        batch_idx: int,
        *args,
        **kwargs,
    ):
        r"""Since Callback.on_batch_end does not provide access to the batch and outputs, we must
        implement on_X_batch_end for each mode and call this method.
        """
        state = pl_module.state
        self.state_metrics.update(state, batch, outputs)

        if self.log_on_step:
            collection = self.state_metrics.get_state(state)
            tag = state.with_postfix(self.name)
            self.wrapped_log(
                collection.compute(),
                pl_module,
                tag,
                trainer.global_step,
            )
            collection.reset()

    def _on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: BaseModel,
        mode: Mode,
    ):
        trainer.global_step
        for state, metric in self.state_metrics.as_dict().items():
            tag = state.with_postfix(self.name)
            if state.mode == mode:
                self.wrapped_log(
                    metric.compute(),
                    pl_module,
                    tag,
                    trainer.global_step,
                )
                metric.reset()


@CALLBACK_REGISTRY
class ErrorAtUncertaintyCallback(MetricLoggingCallback):
    def __init__(
        self,
        name: str,
        modes: Iterable[Union[str, Mode]] = ALL_MODES,
        log_on_step: bool = False,
        **kwargs,
    ):
        kwargs["from_logits"] = True
        kwargs.setdefault("num_bins", 10)
        metric = MetricCollection({name: ErrorAtUncertainty(**kwargs)})
        super().__init__(name, metric, modes, log_on_step)

    def log_target(
        self,
        target: Dict[str, Any],
        pl_module: BaseModel,
        tag: str,
        step: int,
    ):
        entropy, err, totals = next(iter(target.values()))
        entropy = entropy[totals.bool()]
        err = err[totals.bool()]
        fig = ErrorAtUncertainty.plot(entropy, err)
        pl_module.wrapped_log({tag: wandb.Image(fig)})


@CALLBACK_REGISTRY
class ConfusionMatrixCallback(MetricLoggingCallback[I, O]):
    def __init__(
        self,
        name: str,
        num_classes: int,
        modes: Iterable[Union[str, Mode]] = ALL_MODES,
        log_on_step: bool = False,
        **kwargs,
    ):
        metric = MetricCollection({name: ConfusionMatrix(num_classes, **kwargs)})
        super().__init__(name, metric, modes, log_on_step)

    def log_target(
        self,
        target: Dict[str, Any],
        pl_module: BaseModel,
        tag: str,
        step: int,
    ):
        mat = next(iter(target.values()))
        fig = ConfusionMatrix.plot(mat)
        pl_module.wrapped_log({tag: wandb.Image(fig)})
