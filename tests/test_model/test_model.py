#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

from project.model import BaseModel, ConvModel
from project.structs import Example


class DummyDataset(Dataset):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __getitem__(self, index):
        img = torch.rand(3, 32, 32)
        label = torch.randint(0, self.num_classes, (1,)).squeeze_()
        return img, label

    def __len__(self) -> int:
        return 100


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def train_dataloader(self, *args, **kwargs):
        return self._dataloader()

    def val_dataloader(self, *args, **kwargs):
        return self._dataloader()

    def test_dataloader(self, *args, **kwargs):
        return self._dataloader()

    def _dataloader(self):
        ds = DummyDataset(self.num_classes)
        return DataLoader(ds, batch_size=4, num_workers=4)


OPTIMIZER = {"class_path": "torch.optim.Adam", "init_args": {"lr": 0.001}}


@pytest.mark.parametrize("model", [ConvModel(optimizer_init=OPTIMIZER)])
def test_init_train(tmpdir, model: BaseModel):
    datamodule: pl.LightningDataModule = DummyDataModule(num_classes=10)
    logger = WandbLogger(project="test_init_train", save_dir=str(tmpdir), offline=True)

    steps = 10000
    trainer = pl.Trainer(
        logger=logger,
        min_steps=steps,
        max_steps=steps,
        default_root_dir=tmpdir,
        fast_dev_run=5,
        log_every_n_steps=2,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


@pytest.mark.parametrize(
    "model,raw",
    [
        pytest.param(
            ConvModel(),
            (
                torch.rand(2, 3, 32, 32),
                torch.rand(
                    2,
                ),
            ),
        ),
        pytest.param(
            ConvModel(),
            Example(
                torch.rand(2, 3, 32, 32),
                torch.rand(
                    2,
                ),
            ),
        ),
    ],
)
def test_wrap_example(model, raw):
    wrapped = model.wrap_example(raw)
    assert isinstance(wrapped, Example)
