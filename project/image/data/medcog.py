#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from typing import Any, Dict, List, TypedDict
from pathlib import Path
from argparse import ArgumentParser

from flash.core.data.io.input import DataKeys, Input
from flash.core.data.utilities.paths import PATH_TYPE, filter_valid_files
from flash.core.data.utilities.samples import to_sample
from flash.core.data.utils import image_default_loader
from flash.core.utilities.stages import RunningStage



def medcog_load(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    example = torch.load(path)
    import pdb; pdb.set_trace()


class MedcogInput(Input):
    def load_data(self, file: PATH_TYPE) -> List[Dict[str, Any]]:
        file = Path(file)
        files = filter_valid_files(list(file.glob("*.pth")), valid_extensions=(".pth",))
        if not files:
            raise FileNotFoundError(file)
        return [to_sample(f) for f in files]

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        filepath = sample[DataKeys.INPUT]
        data = medcog_load(filepath)
        import pdb; pdb.set_trace()
        sample[DataKeys.INPUT] = image_default_loader(filepath)
        sample[DataKeys.TARGET] = ...
        sample.setdefault(DataKeys.METADATA, {})
        sample[DataKeys.METADATA]["filepath"] = filepath
        sample[DataKeys.METADATA]["shape"] = sample[DataKeys.INPUT].shape[-2:]
        return sample



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("path", type=Path, help="path to preprocessed data subdir")
    return parser.parse_args()

def main():
    args = parse_args()
    inp = MedcogInput(RunningStage.TRAINING, args.path)
    ex = inp[0]
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
