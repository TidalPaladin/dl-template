#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datasets import DummyDataset


class CIFAR10DummyDataset(DummyDataset):
    def __init__(self, *args, **kwargs):
        img_shape = (3, 32, 32)
        super().__init__(img_shape)

    def __getitem__(self, idx: int):
        img = super().__getitem__(idx)[0]
        label = torch.randint(0, 10, (1,)).squeeze()
        return img, label


class DummyDataModule(CIFAR10DataModule):
    dataset_cls = CIFAR10DummyDataset
