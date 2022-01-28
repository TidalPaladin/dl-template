#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pl_bolts
from jsonargparse import lazy_instance
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.cli import (
    DATAMODULE_REGISTRY,
    LR_SCHEDULER_REGISTRY,
    MODEL_REGISTRY,
    OPTIMIZER_REGISTRY,
    LightningArgumentParser,
    LightningCLI,
)

from project import model

from .model import BaseModel, ConvModel


MODEL_REGISTRY.register_classes(model, BaseModel, override=True)
DATAMODULE_REGISTRY.register_classes(pl_bolts.datamodules, VisionDataModule, override=True)

PROJECT = "cifar10"
OUTPUT = os.environ.get("OUTPUT_PATH", "./outputs")
OUTPUT += f"/{PROJECT}"


TRAINER_DEFAULTS = {
    "default_root_dir": "./outputs",
}


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.add_optimizer_args(OPTIMIZER_REGISTRY.classes, link_to="model.init_args.optimizer_init")
        parser.add_lr_scheduler_args(LR_SCHEDULER_REGISTRY.classes, link_to="model.init_args.lr_scheduler_init")
        parser.add_argument("--lr_scheduler_monitor", default="train/total_loss_epoch")
        parser.add_argument("--lr_scheduler_interval", default="epoch")

        parser.link_arguments("lr_scheduler_monitor", "model.init_args.lr_scheduler_monitor")
        parser.link_arguments("lr_scheduler_interval", "model.init_args.lr_scheduler_interval")

        parser.set_defaults(
            {
                "trainer.logger": lazy_instance(WandbLogger, project=PROJECT, log_model=True, save_dir=OUTPUT),
                "model": lazy_instance(ConvModel),
                "data": lazy_instance(CIFAR10DataModule, data_dir="./data", num_workers=8, batch_size=32),
                "optimizer": "torch.optim.AdamW",
            }
        )


def main():
    print(f"Writing to {OUTPUT}")
    cli = CLI(
        BaseModel,
        VisionDataModule,
        seed_everything_default=42,
        trainer_defaults=TRAINER_DEFAULTS,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_filename=f"{PROJECT}/config.yaml",
        save_config_overwrite=True,
    )


if __name__ == "__main__":
    main()
