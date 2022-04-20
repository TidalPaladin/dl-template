#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Type, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from flash.core.data.base_viz import BaseVisualization
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule, DatasetInput
from flash.core.data.io.input import DataKeys, Input
from flash.core.data.io.input_transform import INPUT_TRANSFORM_TYPE
from flash.core.data.utilities.paths import PATH_TYPE
from flash.core.integrations.labelstudio.input import _parse_labelstudio_arguments, LabelStudioImageClassificationInput
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import (
    _FIFTYONE_AVAILABLE,
    _IMAGE_EXTRAS_TESTING,
    _IMAGE_TESTING,
    _MATPLOTLIB_AVAILABLE,
    Image,
    requires,
)
from flash.core.utilities.stages import RunningStage
from flash.image.classification.input import (
    ImageClassificationFilesInput,
    ImageClassificationFolderInput,
    ImageClassificationTensorInput,
)
from flash.image.classification.input_transform import ImageClassificationInputTransform
from ..data.input import MedcogFilesInput, MedcogFolderInput


class BreastData(DataModule):
    """The ``BreastData`` class is a :class:`~flash.core.data.data_module.DataModule` with a set of
    classmethods for loading data for image classification."""

    input_transforms_registry = FlashRegistry("input_transforms")
    input_transform_cls = ImageClassificationInputTransform

    @classmethod
    def from_files(
        cls,
        train_files: Optional[Sequence[str]] = None,
        train_targets: Optional[Sequence[Any]] = None,
        val_files: Optional[Sequence[str]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_files: Optional[Sequence[str]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_files: Optional[Sequence[str]] = None,
        train_transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        input_cls: Type[Input] = MedcogFilesInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "BreastData":
        ds_kw = dict(
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        train_input = input_cls(RunningStage.TRAINING, train_files, train_targets, transform=train_transform, **ds_kw)
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_files, val_targets, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_files, test_targets, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_files, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        train_transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        input_cls: Type[Input] = MedcogFolderInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "BreastData":
        ds_kw = dict(
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        train_input = input_cls(RunningStage.TRAINING, train_folder, transform=train_transform, **ds_kw)
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_folder, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_folder, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_folder, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        train_transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        input_cls: Type[Input] = DatasetInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "DataModule":
        ds_kw = dict(
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_dataset, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_dataset, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_dataset, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_dataset, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @classmethod
    def from_preprocessed(
        cls,
        train_folders: List[str] = [],
        val_folders: List[str] = [],
        test_folders: List[str] = [],
        train_transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        input_cls: Type[Input] = MedcogFolderInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "BreastData":
        ds_kw = dict(
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        train_input = input_cls(RunningStage.TRAINING, train_folder, transform=train_transform, **ds_kw)
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_folder, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_folder, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, None, transform=ImageClassificationInputTransform, **ds_kw),
            **data_module_kwargs,
        )
