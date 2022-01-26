#!/usr/bin/env python
# -*- coding: utf-8 -*-

import runpy
import sys

import pytest


@pytest.mark.skip
def test_fast_dev_run():
    sys.argv = [sys.argv[0], "trainer=test"]
    runpy.run_module("src.project", run_name="__main__", alter_sys=True)


@pytest.mark.skip
def test_dev_run():
    sys.argv = [sys.argv[0], "trainer=test", "trainer.params.fast_dev_run=False"]
    runpy.run_module("src.project", run_name="__main__", alter_sys=True)
