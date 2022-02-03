#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import pytest

from project.callbacks import TableCallback
from project.structs import Example, Prediction, State
from tests.test_callbacks.base_callback import BaseCallbackTest


class MyCallback(TableCallback):
    def __init__(self, *args, **kwargs):
        data = {
            "sum": [0],
            "p": [0],
        }
        proto = pd.DataFrame(data)
        super().__init__(*args, **kwargs, proto=proto)

    def create_table(self, state: State, example: Example, pred: Prediction) -> pd.DataFrame:
        data = {
            "sum": [example.img.sum().item()],
            "p": [pred.probs.sum().item()],
        }
        return pd.DataFrame(data)


# TODO finish building these tests
class TestTableCallback(BaseCallbackTest):
    @pytest.fixture
    def callback(self, modes):
        cb = MyCallback("table", modes=modes)
        return cb
