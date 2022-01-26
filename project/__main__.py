#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities.cli import LightningCLI
from pl_bolts.datamodules import CIFAR10DataModule
from functools import partial
from .model import ConvModel, BaseModel
from pytorch_lightning.plugins import DDPPlugin, SingleDevicePlugin
from typing import Union, List


def patch_logger(logger: WandbLogger) -> WandbLogger:
    log = logger.experiment.log
    logger.experiment.log = partial(log, commit=False)
    return logger


def main():
    pl.seed_everything(42)
    model: BaseModel = ConvModel()
    datamodule: pl.LightningDataModule = CIFAR10DataModule(data_dir="./data", num_workers=24, batch_size=64)
    ROOT: Path = Path("/mnt/iscsi/outputs/cifar10")
    logger = WandbLogger(project="cifar10", log_model=True, save_dir=str(ROOT))
    callbacks = model.get_callbacks()

    logger = patch_logger(logger)

    gpus: Union[int, List[int]] = [0]

    if isinstance(gpus, int) and gpus > 1 or isinstance(gpus, list) and len(gpus) > 1:
        strategy = DDPPlugin(find_unused_parameters=True)
    else:
        strategy = None

    print(f"Artifacts will be saved to {str(ROOT)}")

    steps = 40000
    trainer = pl.Trainer(
        logger=logger,
        precision=16,
        gpus=gpus,
        min_steps=steps,
        max_steps=steps,
        callbacks=callbacks,
        accumulate_grad_batches=1,
        num_sanity_val_steps=0,
        strategy=strategy,
        check_val_every_n_epoch=1,
        gradient_clip_val=10.0,
        default_root_dir=ROOT,
    )
    
    trainer.fit(model, datamodule=datamodule)



if __name__ == "__main__":
    main()
