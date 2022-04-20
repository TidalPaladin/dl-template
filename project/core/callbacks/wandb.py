#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional, Set

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only


class WandBSaveCallback(Callback):
    def __init__(
        self,
        pattern: str,
        path: Optional[Path] = None,
        policy: str = "end",
    ):
        assert policy in ("live", "now", "end")
        self.pattern = pattern
        self.policy = policy
        self._path = path

    @rank_zero_only
    def save(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for path in self.get_search_paths(trainer, pl_module):
            assert isinstance(pl_module.logger, LightningLoggerBase)
            pl_module.logger.experiment.save(
                str(Path(path, "*.ckpt")),
                policy=self.policy,
            )

    def get_search_paths(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> Set[Path]:
        if self._path is not None:
            return {self._path}

        assert isinstance(trainer.logger, LightningLoggerBase)
        root = trainer.logger.save_dir or trainer.default_root_dir
        version = (
            trainer.logger.version if isinstance(trainer.logger.version, str) else f"version_{trainer.logger.version}"
        )
        path = Path(root, str(trainer.logger.name), version)
        return {path}


class WandBCheckpointCallback(WandBSaveCallback):
    def __init__(
        self,
        pattern: str = "*.ckpt",
        policy: str = "end",
    ):
        super().__init__(pattern, None, policy)

    def get_search_paths(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> Set[Path]:
        paths: Set[Path] = set()
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.dirpath:
                paths.add(Path(cb.dirpath))
        return paths

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.save(trainer, pl_module)

    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *_):
        self.save(trainer, pl_module)
