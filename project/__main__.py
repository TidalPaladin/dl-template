#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities.cli import LightningCLI


def get_callbacks():
    return [LearningRateMonitor()]


def main():
    model = ...
    datamodule = ...
    ROOT = Path(...)
    logger = WandbLogger(project="mammogram", log_model=True, save_dir=ROOT)

    steps = 10000
    trainer = pl.Trainer(
        logger=logger,
        precision=16,
        gpus=2,
        min_steps=steps,
        max_steps=steps,
        callbacks=get_callbacks(),
        accumulate_grad_batches=1,
        num_sanity_val_steps=0,
        strategy="ddp",
        check_val_every_n_epoch=1,
        gradient_clip_val=10.0,
        default_root_dir=ROOT,
        #resume_from_checkpoint="fcos_no_utsw.ckpt",
    )
    
    trainer.fit(model, datamodule=datamodule)



if __name__ == "__main__":
    main()
