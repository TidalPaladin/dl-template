#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
from functools import partial
from typing import Any, ClassVar, Dict, Generic, Iterable, Iterator, Optional, Tuple, cast

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore
from torchmetrics import MetricCollection

from combustion.util import MISSING

from ..callbacks import ConfusionMatrixCallback, ErrorAtUncertaintyCallback, QueuedImageLoggingCallback
from ..data import NamedDataModuleMixin
from ..metrics import UCE, Accuracy, Entropy, MetricStateCollection
from ..structs import Example, I, L, Loss, Mode, O, Prediction, State


class BaseModel(pl.LightningModule, Generic[I, O, L]):
    r"""Base class for all models."""
    state: State
    example_type: ClassVar[Example] = Example
    logger: LightningLoggerBase

    def __init__(self, lr: float = 1e-3, weight_decay: float = 0, num_classes: int = 10):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.state = State()
        self._batch_size = 4
        self.num_classes = num_classes
        self.state_metrics = MetricStateCollection()

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
            "uce": UCE(num_bins=10, num_classes=10, from_logits=True),
        }
        return MetricCollection(metrics, prefix=state.prefix).to(self.device)

    def get_callbacks(self) -> Iterable[Callback]:
        Q = 8
        return [
            LearningRateMonitor(),
            QueuedImageLoggingCallback("image_worst", queue_size=Q, modes=[Mode.VAL, Mode.TEST]),
            QueuedImageLoggingCallback("image_worst", queue_size=Q, modes=[Mode.TRAIN], flush_interval=1000),
            QueuedImageLoggingCallback("image_best", queue_size=Q, modes=[Mode.VAL, Mode.TEST], negate_priority=True),
            QueuedImageLoggingCallback(
                "image_best", queue_size=Q, modes=[Mode.TRAIN], flush_interval=1000, negate_priority=True
            ),
            ErrorAtUncertaintyCallback("uncert", modes=[Mode.VAL, Mode.TRAIN, Mode.TEST]),
            ConfusionMatrixCallback("conf_mat", modes=[Mode.VAL, Mode.TEST], num_classes=self.num_classes),
        ]

    def log_loss(self, loss: L) -> None:
        name = self.state.with_postfix("total_loss")
        self._log_loss(name, loss.total_loss)

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), self.lr, weight_decay=self.weight_decay)
        scheduler = OneCycleLR(
            opt,
            self.lr,
            total_steps=self.trainer.max_steps,
            pct_start=0.3,
            div_factor=1,
            final_div_factor=20,
        )
        schedule_dict = {"scheduler": scheduler, "interval": "step", "name": "train/lr"}
        return [opt], [schedule_dict]

    def compute_loss(self, example: I, pred: O) -> L:
        assert example.label is not None
        loss = Loss(F.cross_entropy(pred.logits, example.label))
        return cast(L, loss)

    def step(self, example: I, batch_idx: int, *args) -> Tuple[O, L]:
        # forward pass
        pred: O = self(example)
        assert isinstance(pred, Prediction), "model should return a Prediction instance"

        # compute loss
        loss = self.compute_loss(example, pred)
        assert isinstance(loss, Loss), "compute_loss should return a Loss instance"

        # update metrics
        self.state_metrics.update(self.state, example, pred)

        # loss and metric logging
        self.log_loss(loss)
        self.state_metrics.log(self.state, self)

        return pred, loss

    def training_step(self, example: I, batch_idx: int, *args):
        pred, loss = self.step(example, batch_idx)
        return {"pred": pred.detach(), "loss": loss.total_loss}

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
        for mode in (Mode.TRAIN, Mode.VAL):
            for name in self.dataset_names(mode):
                state = State(mode, name if mode == Mode.VAL else None)
                metrics = self.get_metrics(state).to(self.device)
                self.state_metrics.set_state(state, metrics)

        # TODO: this doesn't print for some reason
        self.print("Metrics:")
        self.print(self.state_metrics.summarize())

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

    def commit_logs(self, step: int = None) -> None:
        if isinstance(self.logger, WandbLogger):
            # final log call with commit=True to flush results
            self.logger.experiment.log({}, commit=True, step=step)
        # ensure all pyplot plots are closed
        plt.close()

    @rank_zero_only
    def wrapped_log(self, items: Dict[str, Any]):
        target = {"trainer/global_step": self.trainer.global_step}
        target.update(items)
        self.logger.experiment.log(target, commit=False)

    def get_dataset_name(self, mode: Mode, dataloader_idx: Optional[int] = None) -> Optional[str]:
        if not hasattr(self.trainer, "datamodule"):
            return None
        elif isinstance((dm := self.trainer.datamodule), NamedDataModuleMixin):
            return dm.get_name(mode, dataloader_idx)
        elif hasattr(dm, "name"):
            return self.trainer.datamodule.name
        else:
            return self.trainer.datamodule.__class__.__name__

    def dataset_names(self, mode: Mode) -> Iterator[str]:
        if not hasattr(self.trainer, "datamodule"):
            return
        elif isinstance((dm := self.trainer.datamodule), NamedDataModuleMixin):
            for name in dm.names_for_mode(mode):
                yield name
        elif hasattr(dm, "name"):
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
            f = partial(log, commit=False, step=self.global_step)
            return f(*args, **kwargs)

        logger.experiment.log == wrapped_log
        return logger
