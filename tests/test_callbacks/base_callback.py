#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
from pytorch_lightning.loggers import LightningLoggerBase

from project.callbacks import LoggingCallback
from project.model import BaseModel
from project.structs import Example, Mode, MultiClassPrediction, Prediction, State


class BaseCallbackTest:
    @pytest.fixture(
        params=[
            [Mode.TRAIN],
            [Mode.VAL, Mode.TEST],
            [Mode.TEST],
        ]
    )
    def modes(self, request):
        return request.param

    @pytest.fixture
    def callback(self):
        ...

    @pytest.fixture
    def batch_idx(self):
        return 10

    @pytest.fixture
    def example(self):
        torch.random.manual_seed(42)
        img = torch.rand(4, 3, 32, 32)
        label = torch.randint(0, 10, (4, 1))
        return Example(img, label)

    @pytest.fixture
    def prediction(self):
        torch.random.manual_seed(42)
        logits = torch.rand(4, 10)
        return MultiClassPrediction(logits)

    def on_train_batch_end(
        self,
        callback: LoggingCallback,
        pl_module: BaseModel,
        logger: LightningLoggerBase,
        output: Prediction,
        example: Example,
        batch_idx: int,
    ):
        trainer = pl_module.trainer
        callback.on_train_batch_end(
            trainer,
            pl_module,
            output,  # type: ignore
            example,
            batch_idx,
        )

        return callback

    def on_validation_batch_end(
        self,
        callback: LoggingCallback,
        pl_module: BaseModel,
        logger: LightningLoggerBase,
        output: Prediction,
        example: Example,
        batch_idx: int,
    ):
        trainer = pl_module.trainer
        callback.on_train_batch_end(
            trainer,
            pl_module,
            output,  # type: ignore
            example,
            batch_idx,
        )

        return callback

    def on_test_batch_end(
        self,
        callback: LoggingCallback,
        pl_module: BaseModel,
        logger: LightningLoggerBase,
        output: Prediction,
        example: Example,
        batch_idx: int,
    ):
        trainer = pl_module.trainer
        callback.on_train_batch_end(
            trainer,
            pl_module,
            output,  # type: ignore
            example,
            batch_idx,
        )

        return callback

    def on_train_epoch_end(
        self,
        callback: LoggingCallback,
        pl_module: BaseModel,
        logger: LightningLoggerBase,
    ):
        trainer = pl_module.trainer
        callback.on_train_epoch_end(
            trainer,
            pl_module,
        )

        return callback

    def on_validation_epoch_end(
        self,
        callback: LoggingCallback,
        pl_module: BaseModel,
        logger: LightningLoggerBase,
    ):
        trainer = pl_module.trainer
        callback.on_train_epoch_end(
            trainer,
            pl_module,
        )

        return callback

    def on_test_epoch_end(
        self,
        callback: LoggingCallback,
        pl_module: BaseModel,
        logger: LightningLoggerBase,
    ):
        trainer = pl_module.trainer
        callback.on_train_epoch_end(
            trainer,
            pl_module,
        )
        return callback

    def test_len(self, callback):
        length = len(callback)
        assert isinstance(length, int)

    @pytest.mark.parametrize(
        "state",
        [
            pytest.param(State(Mode.TRAIN)),
            pytest.param(State(Mode.VAL, "foo")),
            pytest.param(State(Mode.TEST, "bar")),
        ],
    )
    def test_register(self, callback, pl_module, state, example, prediction):
        callback.register(state, pl_module, example, prediction)

    def test_reset(self, callback):
        callback.reset()
        assert len(callback) == 0

    @pytest.mark.parametrize("mode", [Mode.TRAIN, Mode.VAL, Mode.TEST])
    def test_reset_on_epoch_begin(self, mocker, pl_module, callback, mode):
        spy = mocker.spy(callback, "reset")
        trainer = pl_module.trainer
        if mode == Mode.TRAIN:
            callback.on_train_epoch_begin(trainer, pl_module)
        elif mode == Mode.VAL:
            callback.on_validation_epoch_begin(trainer, pl_module)
        elif mode == Mode.TEST:
            callback.on_test_epoch_begin(trainer, pl_module)
        else:
            raise ValueError()
        spy.assert_called_once_with(specific_modes=[mode])

    @pytest.mark.parametrize("mode", [Mode.TRAIN, Mode.VAL, Mode.TEST])
    def test_no_log_on_sanity_check(self, pl_module, callback, mode, example, prediction, batch_idx):
        trainer = pl_module.trainer
        logger = pl_module.logger
        pl_module.state = State(mode)
        callback.on_sanity_check_start(trainer, pl_module)
        if mode == Mode.TRAIN:
            callback.on_train_epoch_begin(trainer, pl_module)
            self.on_train_batch_end(callback, pl_module, logger, prediction, example, batch_idx)
            callback.on_train_epoch_end(trainer, pl_module)
        elif mode == Mode.VAL:
            callback.on_validation_epoch_begin(trainer, pl_module)
            self.on_validation_batch_end(callback, pl_module, logger, prediction, example, batch_idx)
            callback.on_validation_epoch_end(trainer, pl_module)
        elif mode == Mode.TEST:
            callback.on_test_epoch_begin(trainer, pl_module)
            self.on_test_batch_end(callback, pl_module, logger, prediction, example, batch_idx)
            callback.on_test_epoch_end(trainer, pl_module)
        else:
            raise ValueError()
        logger.experiment.log.assert_not_called()

    @pytest.mark.parametrize("mode", [Mode.TRAIN, Mode.VAL, Mode.TEST])
    def test_log(self, pl_module, callback, mode, example, prediction, batch_idx):
        trainer = pl_module.trainer
        logger = pl_module.logger
        pl_module.state = State(mode)
        callback.modes = [mode]
        if mode == Mode.TRAIN:
            callback.on_train_epoch_begin(trainer, pl_module)
            self.on_train_batch_end(callback, pl_module, logger, prediction, example, batch_idx)
            callback.on_train_epoch_end(trainer, pl_module)
        elif mode == Mode.VAL:
            callback.on_validation_epoch_begin(trainer, pl_module)
            self.on_validation_batch_end(callback, pl_module, logger, prediction, example, batch_idx)
            callback.on_validation_epoch_end(trainer, pl_module)
        elif mode == Mode.TEST:
            callback.on_test_epoch_begin(trainer, pl_module)
            self.on_test_batch_end(callback, pl_module, logger, prediction, example, batch_idx)
            callback.on_test_epoch_end(trainer, pl_module)
        else:
            raise ValueError()
        logger.experiment.log.assert_called()
