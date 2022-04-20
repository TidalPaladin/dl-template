#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from project.callbacks import ConfusionMatrixCallback, ErrorAtUncertaintyCallback
from tests.test_callbacks.base_callback import BaseCallbackTest


class TestConfusionMatrixCallback(BaseCallbackTest):
    @pytest.fixture
    def callback(self, modes):
        cb = ConfusionMatrixCallback("name", 10, modes=modes)
        return cb


class TestErrorAtUncertaintyCallback(BaseCallbackTest):
    @pytest.fixture
    def callback(self, modes):
        cb = ErrorAtUncertaintyCallback("name", modes)
        return cb
