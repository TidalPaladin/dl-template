#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import pytest
from pathlib import Path
from project.breast.data.input import MedcogFilesInput, MedcogFoldersInput
from flash.core.utilities.stages import RunningStage
from typing import List

@pytest.fixture
def example():
    img = torch.rand(1, 512, 512)
    regional_label = torch.rand(10, 6)
    global_label = torch.rand(3)
    return img, (global_label, regional_label)

@pytest.fixture
def files_factory(example):
    def func(path: Path, num_malign: int = 1, num_benign: int = 1, num_unknown: int = 1):
        names = {
            "malign": num_malign,
            "benign": num_benign,
            "unknown": num_unknown,
        }
        files: List[Path] = []

        for name, limit in names.items():
            subdir = Path(path, name)
            subdir.mkdir()
            for i in range(limit):
                fp = Path(subdir, f"mcn_{i}.pth")
                torch.save(example, fp)
                files.append(fp)
        return files
    return func


class TestFilesInput:

    def test_getitem(self, tmp_path, files_factory):
        path = Path(tmp_path, "data")
        path.mkdir()
        files = files_factory(path)
        i = MedcogFilesInput(RunningStage.TRAINING, files)
        out = i[0]
        assert False
