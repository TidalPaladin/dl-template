#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn as nn
from flash import Task as FlashTask
from flash.core.model import OutputKeys
from flash.core.utilities.apply_func import get_callable_dict
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import LOSS_FN_TYPE, METRICS_TYPE, MODEL_TYPE, OUTPUT_TRANSFORM_TYPE
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from registry import Registry
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore

from .io import OUTPUT_REGISTRY, OutputTransform


MODEL_REGISTRY = Registry("models")
BACKBONE_REGISTRY = Registry("backbones")
HEAD_REGISTRY = Registry("heads")
OPTIMIZER_REGISTRY = Registry("optimizers")
LR_SCHEDULER_REGISTRY = Registry("lr_schedulers")
LOSS_FN_REGISTRY = Registry("loss")
TASK_REGISTRY = Registry("tasks")


class WandBMixin:
    r"""Mixin for clean WandB logging"""
    logger: Any
    global_step: int
    trainer: pl.Trainer

    def on_train_batch_end(self, *args, **kwargs):
        self.commit_logs(step=self.global_step)

    @rank_zero_only
    def commit_logs(self, step: Optional[int] = None) -> None:
        if isinstance(self.logger, WandbLogger):
            assert self.global_step >= self.logger.experiment.step

            # final log call with commit=True to flush results
            self.logger.experiment.log.log({}, commit=True, step=self.global_step)
        # ensure all pyplot plots are closed
        plt.close()

    @rank_zero_only
    def wrapped_log(self, items: Dict[str, Any]):
        target = {"trainer/global_step": self.trainer.global_step}
        target.update(items)
        self.logger.experiment.log(target, commit=False)

    def setup(self, *args, **kwargs):
        if isinstance(self.logger, WandbLogger):
            self.patch_logger(self.logger)

    def patch_logger(self, logger: WandbLogger) -> WandbLogger:
        r""":class:`WandbLogger` doesn't expect :func:`log` to be called more than a few times per second.
        Additionally, each call to :func:`log` will increment the logger step counter, which results
        in the logged step value being out of sync with ``self.global_step``. This method patches
        a :class:`WandbLogger` to log using ``self.global-step`` and never commit logs. Logs must be commited
        manually (already implemented in :func:`on_train_batch_end`).
        """
        # TODO provide a way to unpatch the logger (probably needed for test/inference)
        log = logger.experiment.log

        def wrapped_log(*args, **kwargs):
            assert self.global_step >= self.logger.experiment.step
            f = partial(log, commit=False)
            kwargs.pop("commit", None)
            return f(*args, **kwargs)

        # attach the original log method as an attribute of the wrapper
        # this allows commiting logs with logger.experiment.log.log(..., commit=True)
        wrapped_log.log = log

        # apply the patch
        logger.experiment.log = wrapped_log
        return logger


class Task(FlashTask, WandBMixin):
    r"""Base task"""
    logger: LightningLoggerBase
    outputs: Registry = OUTPUT_REGISTRY

    required_extras: Optional[Union[str, List[str]]] = None

    def __init__(
        self,
        model: MODEL_TYPE,
        loss_fn: LOSS_FN_TYPE = None,
        learning_rate: Optional[float] = None,
        metrics: METRICS_TYPE = None,
        output_transform: OUTPUT_TRANSFORM_TYPE = None,
    ):
        super().__init__()
        if model is not None:
            self.model = model
        self.loss_fn = {} if loss_fn is None else get_callable_dict(loss_fn)

        self.train_metrics = nn.ModuleDict({} if metrics is None else get_callable_dict(metrics))
        self.val_metrics = nn.ModuleDict({} if metrics is None else get_callable_dict(deepcopy(metrics)))
        self.test_metrics = nn.ModuleDict({} if metrics is None else get_callable_dict(deepcopy(metrics)))
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self._output_transform: Optional[OutputTransform] = output_transform

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        output = self.step(batch, batch_idx, self.train_metrics)
        log_kwargs = {"batch_size": output.get(OutputKeys.BATCH_SIZE, None)}
        self.log_dict(
            {f"train/{k}": v for k, v in output[OutputKeys.LOGS].items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            **log_kwargs,
        )
        return output

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        output = self.step(batch, batch_idx, self.val_metrics)
        log_kwargs = {"batch_size": output.get(OutputKeys.BATCH_SIZE, None)}
        self.log_dict(
            {f"val/{k}": v for k, v in output[OutputKeys.LOGS].items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            **log_kwargs,
        )
        return output

    def test_step(self, batch: Any, batch_idx: int) -> None:
        output = self.step(batch, batch_idx, self.test_metrics)
        log_kwargs = {"batch_size": output.get(OutputKeys.BATCH_SIZE, None)}
        self.log_dict(
            {f"test/{k}": v for k, v in output[OutputKeys.LOGS].items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            **log_kwargs,
        )
        return output


class MultiTask(FlashTask, ABC):
    def __init__(self, **tasks: FlashTask):
        self.tasks = tasks

    def training_step(self, batch: Any, *args, **kwargs) -> Any:
        outputs: Dict[str, Any] = {}
        for name, task in self.tasks.items():
            self.prepare_input(RunningStage.TRAINING, task, batch, *args, **kwargs)
            outputs[name] = task.training_step(batch, *args, **kwargs)
        output = self.merge_outputs(RunningStage.TRAINING, **outputs)
        return output

    def validation_step(self, batch: Any, *args, **kwargs):
        outputs: Dict[str, Any] = {}
        for name, task in self.tasks.items():
            self.prepare_input(RunningStage.VALIDATING, task, batch, *args, **kwargs)
            outputs[name] = task.validation_step(batch, *args, **kwargs)
        output = self.merge_outputs(RunningStage.VALIDATING, **outputs)
        return output

    def test_step(self, batch: Any, *args, **kwargs):
        outputs: Dict[str, Any] = {}
        for name, task in self.tasks.items():
            self.prepare_input(RunningStage.TESTING, task, batch, *args, **kwargs)
            outputs[name] = task.test_step(batch, *args, **kwargs)
        output = self.merge_outputs(RunningStage.TESTING, **outputs)
        return output

    def predict_step(self, batch: Any, *args, **kwargs):
        outputs: Dict[str, Any] = {}
        for name, task in self.tasks.items():
            self.prepare_input(RunningStage.PREDICTING, task, batch, *args, **kwargs)
            outputs[name] = task.predict_step(batch, *args, **kwargs)
        output = self.merge_outputs(RunningStage.PREDICTING, **outputs)
        return output

    @abstractmethod
    def prepare_input(self, stage: RunningStage, task: FlashTask, batch: Any, *args, **kwargs) -> Any:
        ...

    @abstractmethod
    def merge_outputs(self, stage: RunningStage, **outputs) -> Dict[OutputKeys, Any]:
        ...
