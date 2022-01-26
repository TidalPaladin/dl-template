#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import torch
import pytorch_lightning
import pytorch_lightning as pl
import torchmetrics
import pl_bolts
from typing import Final

from combustion.testing import cuda_or_skip as cuda_or_skip_mark
from combustion.testing.utils import cuda_available
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path

LIBRARIES: Final = (
    torch,
    pytorch_lightning,
    torchmetrics,
    pl_bolts,
)


@pytest.fixture(scope="session")
def torch():
    return pytest.importorskip("torch", reason="test requires torch")


@pytest.fixture(scope="session")
def ignite():
    return pytest.importorskip("ignite", reason="test requires ignite")


@pytest.fixture(
    params=[
        pytest.param(False, id="no_cuda"),
        pytest.param(True, marks=cuda_or_skip_mark, id="cuda"),
    ]
)
def cuda(torch, request):
    return request.param


@pytest.fixture(scope="session")
def cuda_or_skip(torch):
    if not cuda_available():
        pytest.skip("test requires cuda")


def pytest_report_header(config):
    s = "Version Information:\n"
    for library in LIBRARIES:
        s += f"{library.__name__} version: {library.__version__}\n"
    return s


@pytest.fixture
def logger(mocker):
    logger = mocker.MagicMock(name="logger", spec_set=WandbLogger)
    return logger

@pytest.fixture
def lightning_module(mocker, logger):
    trainer = pl.Trainer(logger=logger)
    module = pl.LightningModule()
    module.trainer = trainer
    return module
