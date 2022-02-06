#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import TYPE_CHECKING, ForwardRef, Iterable, Optional

import pandas as pd
import pytorch_lightning as pl
import wandb

from combustion.lightning.callbacks import DistributedDataFrame

from ..metrics import DataFrameStateCollection
from ..structs import Example, I, Mode, O, Prediction, State
from .base import LoggingCallback, ModeGroup


if TYPE_CHECKING:
    from ..model.base import BaseModel
else:
    BaseModel = ForwardRef("BaseModel")


class TableCallback(LoggingCallback[I, O]):
    def __init__(
        self,
        name: str,
        proto: Optional[pd.DataFrame] = None,
        modes: ModeGroup = ["val", "test"],
    ):
        super().__init__(name, modes)
        self._tables = DataFrameStateCollection(proto)

    @abstractmethod
    def create_table(self, state: State, example: I, pred: O) -> pd.DataFrame:
        ...

    def __len__(self) -> int:
        # it is difficult to track tables with pending log calls, so just return 0 here
        return 0

    def reset(self, specific_states: Iterable[State] = [], specific_modes: Iterable[Mode] = []):
        self._tables.reset(
            specific_states=list(specific_states),
            specific_modes=list(specific_modes),
        )

    def register(self, state: State, pl_module: BaseModel, example: I, prediction: O) -> None:
        if state in self._tables.states:
            return
        if self._tables._proto is None:
            proto = self.create_table(state, example, prediction)
        else:
            proto = None
        self._tables.register(state, proto)

    def log_target(
        self,
        target: DistributedDataFrame,
        pl_module: BaseModel,
        tag: str,
        step: int,
    ):
        wandb.Table(data=target)
        target_dict = {"trainer/global_step": step, tag: target}
        pl_module.logger.experiment.log(target_dict, commit=False)

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
        if state.mode not in self.modes:
            return
        if not isinstance(outputs, Prediction):
            raise TypeError(f"Expected `outputs` to be type `Prediction`, found {type(outputs)}")
        if not isinstance(batch, Example):
            raise TypeError(f"Expected `batch` to be type `Example`, found {type(batch)}")

        new_table = DistributedDataFrame(self.create_table(state, batch, outputs))
        if state in self._tables.states:
            old_table = self._tables.get_state(state)
            table = DistributedDataFrame(old_table.append(new_table))
        else:
            self._tables.register(state)
            table = new_table

        assert isinstance(table, DistributedDataFrame)
        self._tables.set_state(state, table)

    def _on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: BaseModel,
        mode: Mode,
    ):
        for state, table in self._tables.as_dict().items():
            if state.mode != mode:
                continue
            tag = state.with_postfix(self.name)
            table = table.gather_all()
            self.wrapped_log(table, pl_module, tag, trainer.global_step)
            self._tables.remove_state(state)
