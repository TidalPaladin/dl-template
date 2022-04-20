#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import abstractmethod, ABC
from functools import partial
from typing import Any, ClassVar, Dict, Generic, Iterator, List, Optional, Tuple, Type, cast


import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger
from copy import deepcopy
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.cli import instantiate_class
from torch import Tensor
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore
from torchmetrics import MetricCollection

from flash import Task as FlashTask
from flash.core.adapter import Adapter, AdapterTask
from flash.core.model import OutputKeys
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import (
    INPUT_TRANSFORM_TYPE,
    LOSS_FN_TYPE,
    LR_SCHEDULER_TYPE,
    METRICS_TYPE,
    MODEL_TYPE,
    OPTIMIZER_TYPE,
    OUTPUT_TRANSFORM_TYPE,
)


class WandBMixin:
    r"""Base class for all models."""
    logger: LightningLoggerBase
    global_step: int
    trainer: pl.Trainer

    def on_train_batch_end(self, *args, **kwargs):
        self.commit_logs(step=self.global_step)

    @rank_zero_only
    def commit_logs(self, step: int = None) -> None:
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

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        output = self.step(batch, batch_idx, self.train_metrics)
        self.log_dict(
            {f"train_{k}": v for k, v in output[OutputKeys.LOGS].items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        assert "total_loss" in output
        return output

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        output = self.step(batch, batch_idx, self.val_metrics)
        self.log_dict(
            {f"val_{k}": v for k, v in output[OutputKeys.LOGS].items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return output

    def test_step(self, batch: Any, batch_idx: int) -> None:
        output = self.step(batch, batch_idx, self.test_metrics)
        self.log_dict(
            {f"test_{k}": v for k, v in output[OutputKeys.LOGS].items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return output


class MultiTask(FlashTask, ABC):
    def __init__(self, **tasks: FlashTask):
        self.tasks = tasks

    def training_step(self, batch: Any, *args, **kwargs) -> Any:
        outputs: Dict[str, Any] = {}
        for name, task in self.tasks.items():
            task_batch = self.prepare_input(RunningStage.TRAINING, task, batch, *args, **kwargs)
            outputs[name] = task.training_step(batch, *args, **kwargs)
        output = self.merge_outputs(RunningStage.TRAINING, **outputs)
        return output

    def validation_step(self, batch: Any, *args, **kwargs):
        outputs: Dict[str, Any] = {}
        for name, task in self.tasks.items():
            task_batch = self.prepare_input(RunningStage.VALIDATING, task, batch, *args, **kwargs)
            outputs[name] = task.validation_step(batch, *args, **kwargs)
        output = self.merge_outputs(RunningStage.VALIDATING, **outputs)
        return output

    def test_step(self, batch: Any, *args, **kwargs):
        outputs: Dict[str, Any] = {}
        for name, task in self.tasks.items():
            task_batch = self.prepare_input(RunningStage.TESTING, task, batch, *args, **kwargs)
            outputs[name] = task.test_step(batch, *args, **kwargs)
        output = self.merge_outputs(RunningStage.TESTING, **outputs)
        return output

    def predict_step(self, batch: Any, *args, **kwargs):
        outputs: Dict[str, Any] = {}
        for name, task in self.tasks.items():
            task_batch = self.prepare_input(RunningStage.PREDICTING, task, batch, *args, **kwargs)
            outputs[name] = task.predict_step(batch, *args, **kwargs)
        output = self.merge_outputs(RunningStage.PREDICTING, **outputs)
        return output

    @abstractmethod
    def prepare_input(self, stage: RunningStage, task: FlashTask, batch: Any, *args, **kwargs) -> Any:
        ...

    @abstractmethod
    def merge_outputs(self, stage: RunningStage, **outputs) -> Dict[OutputKeys, Any]:
        ...
