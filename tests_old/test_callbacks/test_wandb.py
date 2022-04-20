#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
from pytorch_lightning.callbacks import ModelCheckpoint

from project.callbacks import WandBCheckpointCallback, WandBSaveCallback


class TestWandBSaveCallback:
    def test_get_search_paths(self, tmp_path, pl_module):
        trainer = pl_module.trainer
        logger = pl_module.logger
        logger.save_dir = "save_dir"
        logger.name = "name"
        logger.version = "version"
        cb = WandBSaveCallback("*.pth")
        out = cb.get_search_paths(trainer, pl_module)

        assert isinstance(out, set)
        p = next(iter(out))
        assert p == Path(logger.save_dir, logger.name, logger.version)


class TestWandBCheckpointCallback:
    @pytest.mark.parametrize("policy", ["live", "now", "end"])
    def test_on_fit_end(self, tmp_path, pl_module, policy):
        trainer = pl_module.trainer
        callbacks = [
            ModelCheckpoint(str(Path(tmp_path, "foo"))),
            ModelCheckpoint(str(Path(tmp_path, "bar"))),
        ]
        patterns = set(str(Path(x.dirpath, "*.ckpt")) for x in callbacks)
        trainer.callbacks = callbacks

        cb = WandBCheckpointCallback(policy=policy)
        cb.on_fit_end(trainer, pl_module)

        exp = pl_module.logger.experiment
        save = exp.save

        assert save.call_count == len(callbacks)
        for call in save.mock_calls:
            assert call.args[0] in patterns
            assert call.kwargs["policy"] == policy

    @pytest.mark.parametrize("policy", ["live", "now", "end"])
    def test_on_exception(self, tmp_path, pl_module, policy):
        trainer = pl_module.trainer
        callbacks = [
            ModelCheckpoint(str(Path(tmp_path, "foo"))),
            ModelCheckpoint(str(Path(tmp_path, "bar"))),
        ]
        patterns = set(str(Path(x.dirpath, "*.ckpt")) for x in callbacks)
        trainer.callbacks = callbacks

        cb = WandBCheckpointCallback(policy=policy)
        cb.on_exception(trainer, pl_module, ...)

        exp = pl_module.logger.experiment
        save = exp.save

        assert save.call_count == len(callbacks)
        for call in save.mock_calls:
            assert call.args[0] in patterns
            assert call.kwargs["policy"] == policy
