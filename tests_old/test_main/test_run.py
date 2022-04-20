#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import runpy
import sys
from pathlib import Path

import pytest

import project
from combustion.testing import cuda_or_skip


@pytest.fixture
def project_root():
    return Path(project.__file__).parents[1].absolute()


@pytest.fixture
def output_path(tmpdir):
    os.environ["OUTPUT_PATH"] = str(tmpdir)
    return str(tmpdir)


def test_print_config():
    sys.argv = [sys.argv[0], "fit", "--print_config"]
    with pytest.raises(SystemExit):
        runpy.run_module("project", run_name="__main__", alter_sys=True)


def test_dummy_dataset(project_root, output_path):
    config_file = Path(project_root, "config.yaml")
    sys.argv = [
        sys.argv[0],
        "fit",
        f"--config={config_file}",
        "--data=project.data.DummyDataModule",
        "--trainer.fast_dev_run=True",
        "--trainer.gpus=null",
        "--trainer.strategy=null",
        "--trainer.precision=32",
    ]
    runpy.run_module("project", run_name="__main__", alter_sys=True)


@pytest.mark.ci_skip
def test_cpu_dev_run(project_root, output_path):
    config_file = Path(project_root, "config.yaml")
    sys.argv = [
        sys.argv[0],
        "fit",
        f"--config={config_file}",
        "--trainer.fast_dev_run=True",
        "--trainer.gpus=null",
        "--trainer.strategy=null",
        "--trainer.precision=32",
    ]
    runpy.run_module("project", run_name="__main__", alter_sys=True)


@cuda_or_skip
@pytest.mark.ci_skip
def test_gpu_dev_run(project_root, output_path):
    config_file = Path(project_root, "config.yaml")
    sys.argv = [
        sys.argv[0],
        "fit",
        f"--config={config_file}",
        "--trainer.fast_dev_run=True",
        "--trainer.gpus=1",
        "--trainer.strategy=null",
    ]
    runpy.run_module("project", run_name="__main__", alter_sys=True)
