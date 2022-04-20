#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flash.core.utilities.flash_cli import FlashCLI
from project.breast.classification.density.model import BreastDensityTask

__all__ = ["image_classification"]


def from_hymenoptera(
    batch_size: int = 4,
    num_workers: int = 0,
    **data_module_kwargs,
) -> ImageClassificationData:
    """Downloads and loads the Hymenoptera (Ants, Bees) data set."""
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "./data")
    return ImageClassificationData.from_folders(
        train_folder="data/hymenoptera_data/train/",
        val_folder="data/hymenoptera_data/val/",
        batch_size=batch_size,
        num_workers=num_workers,
        **data_module_kwargs,
    )


def image_classification():
    """Classify images."""
    cli = FlashCLI(
        ImageClassifier,
        ImageClassificationData,
        default_datamodule_builder=from_hymenoptera,
        #additional_datamodule_builders=[from_movie_posters],
        default_arguments={
            "trainer.max_epochs": 3,
        },
        datamodule_attributes={"num_classes", "labels", "multi_label"},
    )

    cli.trainer.save_checkpoint("image_classification_model.pt")


if __name__ == "__main__":
    image_classification()
