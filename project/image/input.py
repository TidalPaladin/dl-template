#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List

from flash.core.data.io.input import DataKeys, Input
from flash.core.data.utilities.paths import PATH_TYPE, filter_valid_files
from flash.core.data.utilities.samples import to_sample
from flash.core.data.utils import image_default_loader


class ImageInput(Input):
    def load_data(self, file: PATH_TYPE, length: int = 1) -> List[Dict[str, Any]]:
        files = filter_valid_files([file])
        if not files:
            raise FileNotFoundError(file)
        return [to_sample(file)] * length

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        filepath = sample[DataKeys.INPUT]
        sample[DataKeys.INPUT] = image_default_loader(filepath)
        sample[DataKeys.TARGET] = ...
        sample.setdefault(DataKeys.METADATA, {})
        sample[DataKeys.METADATA]["filepath"] = filepath
        sample[DataKeys.METADATA]["shape"] = sample[DataKeys.INPUT].shape[-2:]
        return sample


class DataModueWrapperInput(Input):
    def load_data(self, file: PATH_TYPE, length: int = 1) -> List[Dict[str, Any]]:
        files = filter_valid_files([file])
        if not files:
            raise FileNotFoundError(file)
        return [to_sample(file)] * length

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        filepath = sample[DataKeys.INPUT]
        sample[DataKeys.INPUT] = image_default_loader(filepath)
        sample[DataKeys.TARGET] = ...
        sample.setdefault(DataKeys.METADATA, {})
        sample[DataKeys.METADATA]["filepath"] = filepath
        sample[DataKeys.METADATA]["shape"] = sample[DataKeys.INPUT].shape[-2:]
        return sample
