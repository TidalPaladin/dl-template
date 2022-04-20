#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, cast

import torch
import pandas as pd
from pathlib import Path

from flash.core.data.io.classification_input import ClassificationInputMixin
from flash.core.data.io.input import DataKeys, Input, LightningEnum
from flash.core.data.utilities.classification import MultiBinaryTargetFormatter, TargetFormatter
from flash.core.data.utilities.data_frame import read_csv, resolve_files, resolve_targets
from flash.core.data.utilities.paths import filter_valid_files, make_dataset, PATH_TYPE, list_subdirs
from flash.core.data.utilities.samples import to_samples
from flash.core.integrations.fiftyone.utils import FiftyOneLabelUtilities
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, lazy_import, requires
from flash.image.data import ImageFilesInput, ImageNumpyInput, ImageTensorInput, IMG_EXTENSIONS, NP_EXTENSIONS
#from medcog_preprocessing import from_int16


PTH_EXTENSIONS: Tuple[str, ...] = (".pth", ".pt")

class MedcogTargetKeys(LightningEnum):
    DENSITY = "density"
    MALIGNANCY = "malign"
    ABNORMALCY = "abnorm"
    TRACES = "trace"


class MedcogFilesInput(Input):
    r"""Input over a list of pickled examples"""
    def load_data(self, files: List[PATH_TYPE]) -> List[Dict[str, Any]]:
        files = cast(List[PATH_TYPE], filter_valid_files(files, valid_extensions=PTH_EXTENSIONS))
        return to_samples(files)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        filepath = sample[DataKeys.INPUT]
        img, (global_target, regional_target) = torch.load(filepath)

        target: Dict[MedcogTargetKeys, Any] = {}
        if global_target is not None:
            malign, abnorm, density = global_target
            target[MedcogTargetKeys.MALIGNANCY] = malign
            target[MedcogTargetKeys.ABNORMALCY] = abnorm
            target[MedcogTargetKeys.DENSITY] = density
        if regional_target is not None:
            target[MedcogTargetKeys.TRACES] = regional_target

        sample[DataKeys.INPUT] = img
        sample[DataKeys.TARGET] = target
        sample = super().load_sample(sample)
        sample[DataKeys.METADATA] = {}
        sample[DataKeys.METADATA]["filepath"] = filepath
        return sample


class MedcogFolderInput(MedcogFilesInput):
    r"""Input for a folder containing a pickled examples"""
    def load_data(self, folder: PATH_TYPE) -> List[Dict[str, Any]]:
        files, targets = make_dataset(folder, extensions=PTH_EXTENSIONS)
        return super().load_data(files)


class MedcogDatasetInput(MedcogFolderInput):
    r"""Input for a data from medcog_preprocessing"""
    def load_data(self, folder: PATH_TYPE, balanced: bool = False) -> List[Dict[str, Any]]:
        folder = Path(folder, "torch")
        data: Dict[str, List[Dict[str, Any]]] = {}
        for subgroup in ("malignant", "benign", "unknown"):
            subdir = Path(folder, subgroup)
            if subdir.is_dir():
                data[subgroup] = super().load_data(subdir)

        malign = Path(folder, "malignant")
        benign = Path(folder, "benign")
        unknown = Path(folder, "unknown")

        files, targets = make_dataset(folder, extensions=PTH_EXTENSIONS)
        return super().load_data(files)
