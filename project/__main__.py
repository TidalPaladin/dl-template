#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from torch import Tensor

from combustion.lightning import HydraMixin
from pytorch_lightning.metrics.classification import Accuracy, F1
from pytorch_lightning.metrics import MetricCollection
from .model import FakeModel


@hydra.main(config_path=config_path, config_name=config_name)
def main(cfg):
    combustion.main(cfg)


if __name__ == "__main__":
    main()
