#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
from functools import partial
from typing import Any, ClassVar, Dict, Generic, Iterator, List, Optional, Tuple, Type, cast

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.cli import instantiate_class
from torch import Tensor
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore
from torchmetrics import MetricCollection

from combustion.util import MISSING

from ..callbacks.wandb import WandBCheckpointCallback
from ..data import NamedDataModuleMixin
from ..metrics import UCE, Accuracy, Entropy, MetricStateCollection
from ..structs import Example, I, L, Loss, Mode, O, Prediction, State


class BaseModel(pl.LightningModule, Generic[I, O, L]):
    r"""Base class for all models."""
    state: State
    example_type: ClassVar[Type[Example]] = Example
    logger: LightningLoggerBase

    def __init__(
        self,
        num_classes: int = 10,
        optimizer_init: dict = {},
        lr_scheduler_init: dict = {},
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_monitor: str = "train/total_loss_epoch",
    ):
        super().__init__()
        self.state = State()
        self._batch_size = 4
        self.num_classes = num_classes
        self.state_metrics = MetricStateCollection()
        self.optimizer_init = optimizer_init
        self.lr_scheduler_init = lr_scheduler_init
        self.lr_scheduler_interval = lr_scheduler_interval
        self.lr_scheduler_monitor = lr_scheduler_monitor

    @abstractmethod
    def forward(self, example: I) -> O:
        r"""Forward pass for your model"""
        ...

    def get_metrics(self, state: State) -> MetricCollection:
        r"""Gets a MetricCollection for a given state"""
        # TODO: confirm sync_dist isn't needed for DDP
        # TODO: ensure compute_on_step has no effect
        sync_dist = state.mode != Mode.TRAIN
        metrics = {
            "accuracy": Accuracy(),
            "entropy": Entropy(),
            "uce": UCE(num_bins=10, num_classes=self.num_classes, from_logits=True),
        }
        return MetricCollection(metrics, prefix=state.prefix).to(self.device)

    @torch.no_grad()
    def log_loss(self, loss: L) -> None:
        name = self.state.with_postfix("total_loss")
        self._log_loss(name, loss.total_loss)

    @torch.no_grad()
    def log_metrics(self, example: I, pred: O) -> None:
        self.state_metrics.update(self.state, example, pred)
        self.state_metrics.log(self.state, self)

    def configure_optimizers(self) -> Dict[str, Any]:
        assert self.optimizer_init
        result: Dict[str, Any] = {}
        optimizer = instantiate_class(self.parameters(), self.optimizer_init)
        result["optimizer"] = optimizer

        if self.lr_scheduler_init:
            lr_scheduler: Dict[str, Any] = {}
            scheduler = instantiate_class(optimizer, self.lr_scheduler_init)
            lr_scheduler["scheduler"] = scheduler
            lr_scheduler["monitor"] = self.lr_scheduler_monitor
            lr_scheduler["interval"] = self.lr_scheduler_interval
            result["lr_scheduler"] = lr_scheduler

        return result

    def compute_loss(self, example: I, pred: O) -> L:
        assert example.label is not None
        example.label
        loss = Loss(F.cross_entropy(pred.logits, example.label))
        return cast(L, loss)

    def step(self, example: I, batch_idx: int, *args) -> Tuple[O, L]:
        # forward pass
        pred: O = self(example)
        assert isinstance(pred, Prediction), "model should return a Prediction instance"

        # compute loss
        loss = self.compute_loss(example, pred)
        assert isinstance(loss, Loss), "compute_loss should return a Loss instance"

        # loss and metric logging
        self.log_loss(loss)
        self.log_metrics(example, pred)

        return pred, loss

    def training_step(self, example: I, batch_idx: int, *args):
        pred, loss = self.step(example, batch_idx)
        total_loss = loss.total_loss
        assert total_loss.grad_fn, "training loss should have a grad_fn"
        return {"pred": pred.detach(), "loss": total_loss}

    def validation_step(self, example: I, batch_idx: int, *args):
        pred, loss = self.step(example, batch_idx)
        return pred.detach()

    def test_step(self, example: I, batch_idx: int, *args):
        pred, loss = self.step(example, batch_idx)
        return pred.detach()

    def on_train_epoch_start(self, *args, **kwargs):
        self.state = self.state.update(Mode.TRAIN, None)

    def on_validation_epoch_start(self, *args, **kwargs):
        self.state = self.state.update(Mode.VAL, None)

    def on_test_epoch_start(self, *args, **kwargs):
        self.state = self.state.update(Mode.TEST, None)

    def on_train_batch_start(self, *args, **kwargs):
        r"""Set state before batch"""
        self.state = self.state.update(Mode.TRAIN, None)

    def on_validation_batch_start(self, batch: I, batch_idx: int, dataloader_idx: int):
        r"""Set state before batch"""
        dataset_name = self.get_dataset_name(Mode.VAL, dataloader_idx)
        self.state = self.state.set_dataset(dataset_name).set_mode(Mode.VAL)

    def on_test_batch_start(self, batch: I, batch_idx: int, dataloader_idx: int):
        r"""Set state before batch"""
        dataset_name = self.get_dataset_name(Mode.TEST, dataloader_idx)
        self.state = self.state.set_dataset(dataset_name).set_mode(Mode.TEST)

    def on_fit_start(self):
        r"""Initialize validation/training metrics"""
        # TODO: should we unregister these states in on_fit_end?
        for mode in (Mode.TRAIN, Mode.VAL):
            for name in self.dataset_names(mode):
                state = State(mode, name if mode == Mode.VAL else None)
                metrics = self.get_metrics(state).to(self.device)
                self.state_metrics.set_state(state, metrics)

        # TODO: this doesn't print for some reason
        self.print("Metrics:")
        self.print(self.state_metrics.summarize())
        if isinstance(self.logger, WandbLogger):
            self.patch_logger(self.logger)

    def on_test_start(self):
        r"""Initialize testing metrics"""
        for name in self.dataset_names(Mode.TEST):
            state = State(Mode.TEST, name)
            metrics = self.get_metrics(state).to(self.device)
            self.state_metrics.set_state(state, metrics)

        self.print("Metrics:")
        self.print(self.state_metrics.summarize())

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int):
        r"""Wraps the input batch in an ``Example`` instance if necessary"""
        return self.wrap_example(batch)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int):
        r"""Ensures the input is a batched ``Example``, and captures the batch size"""
        assert isinstance(batch, Example)
        assert batch.is_batched
        # NOTE: PL doesn't pull batch size from dataclasses well, so capture it here
        self._batch_size = len(batch)
        return batch

    def on_train_batch_end(self, *args, **kwargs):
        self.commit_logs(step=self.global_step)

    def wrap_example(self, example: Any) -> I:
        r"""Provides support for existing datasets that do not wrap their examples
        in an :class:`Example`. If ``example_type`` is a defined attribute of the model,
        examples that are not of type :class:`Example` will be converted to :class:`Example`.
        Conversion failure will raise an exception.
        """
        if isinstance(example, Example):
            return cast(I, example)
        elif self.example_type is MISSING:
            raise AttributeError("Please specify the classvar `example_type` to enable autocasting of inputs")

        try:
            return cast(I, self.example_type.create(*example))
        except Exception:
            raise RuntimeError("Failed to create example")

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

    def get_dataset_name(self, mode: Mode, dataloader_idx: Optional[int] = None) -> Optional[str]:
        names = list(self.dataset_names(mode))
        if dataloader_idx is None:
            return names[0] if names else None
        else:
            return names[dataloader_idx]

    def dataset_names(self, mode: Mode) -> Iterator[str]:
        if not hasattr(self.trainer, "datamodule"):
            return
        elif isinstance((dm := self.trainer.datamodule), NamedDataModuleMixin):
            for name in dm.names_for_mode(mode):
                yield name
        elif hasattr(dm, "name") and dm.name != ...:
            yield self.trainer.datamodule.name
        else:
            yield self.trainer.datamodule.__class__.__name__

    def _log_loss(self, name: str, loss: Tensor):
        if self.state.mode == Mode.TRAIN:
            self.log(
                name,
                loss,
                on_step=True,
                on_epoch=False,
                add_dataloader_idx=False,
                batch_size=self._batch_size,
            )
            self.log(
                f"{name}_epoch",
                loss,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
                batch_size=self._batch_size,
            )
        else:
            self.log(
                name,
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                add_dataloader_idx=False,
                batch_size=self._batch_size,
            )

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

    def configure_callbacks(self) -> List[Callback]:
        return [WandBCheckpointCallback()]
