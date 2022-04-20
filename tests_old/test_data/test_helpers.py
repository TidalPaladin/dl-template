#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pytorch_lightning as pl

from project.data import NamedDataModuleMixin
from project.structs import Mode


class DummyDataModule(pl.LightningDataModule, NamedDataModuleMixin):
    def __init__(self, *args, **kwargs):
        self._lookup = {}


class TestNamedDataModuleMixin:
    def test_name(self):
        dm = DummyDataModule()
        assert dm.name == ...

    @pytest.mark.parametrize(
        "mode,name",
        [
            pytest.param(Mode.TEST, "test"),
            pytest.param(Mode.TRAIN, "train"),
        ],
    )
    def test_register_name(self, mode, name):
        dm = DummyDataModule()
        dm.register_name(mode, name)
        assert name in dm.all_names

    @pytest.mark.parametrize(
        "mode,name,dl_index",
        [
            pytest.param(Mode.VAL, "val1", 0),
            pytest.param(Mode.VAL, "val2", 1),
            pytest.param(Mode.VAL, None, 2, marks=pytest.mark.xfail(raises=IndexError)),
            pytest.param(Mode.VAL, None, None, marks=pytest.mark.xfail(raises=ValueError)),
            pytest.param(Mode.TEST, "test", None),
            pytest.param(Mode.TEST, "test", 0),
            pytest.param(Mode.TRAIN, None, None, marks=pytest.mark.xfail(raises=KeyError)),
        ],
    )
    def test_get_name(self, mode, name, dl_index):
        dm = DummyDataModule()
        dm.register_name(Mode.VAL, "val1")
        dm.register_name(Mode.VAL, "val2")
        dm.register_name(Mode.TEST, "test")
        assert dm.get_name(mode, dl_index) == name

    def test_names_for_mode(self):
        dm = DummyDataModule()
        dm.register_name(Mode.VAL, "val1")
        dm.register_name(Mode.VAL, "val2")
        dm.register_name(Mode.TEST, "test")

        assert list(dm.names_for_mode(Mode.VAL)) == ["val1", "val2"]
        assert list(dm.names_for_mode(Mode.TEST)) == ["test"]
        assert list(dm.names_for_mode(Mode.TRAIN)) == []
