#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from project.structs import Mode, State

class TestMode:

    @pytest.mark.parametrize("mode,prefix", [
        pytest.param(Mode.TRAIN, "train"),
        pytest.param(Mode.VAL, "val"),
        pytest.param(Mode.TEST, "test"),
        pytest.param(Mode.INFER, "infer"),
    ])
    def test_prefix(self, mode, prefix):
        assert mode.prefix == prefix

class TestState:

    @pytest.mark.parametrize("mode,dataset", [
        pytest.param(Mode.TRAIN, "train"),
        pytest.param(Mode.TEST, "cifar10"),
        pytest.param(Mode.VAL, "imagenet"),
        pytest.param(Mode.INFER, None),
    ])
    def test_construct(self, mode, dataset):
        state = State(mode, dataset)
        assert state.mode == mode
        assert state.dataset == dataset

    @pytest.mark.parametrize("mode,dataset", [
        pytest.param(Mode.TRAIN, "train"),
        pytest.param(Mode.TEST, "cifar10"),
        pytest.param(Mode.VAL, "imagenet"),
        pytest.param(Mode.INFER, None),
    ])
    def test_update(self, mode, dataset):
        state = State(Mode.INFER, None)
        new_state = state.update(mode, dataset)
        assert new_state.mode == mode
        assert new_state.dataset == dataset

    @pytest.mark.parametrize("mode,dataset,prefix", [
        pytest.param(Mode.TRAIN, "train", "train/"),
        pytest.param(Mode.TRAIN, None, "train/"),
        pytest.param(Mode.TEST, "cifar10", "test/cifar10/"),
        pytest.param(Mode.VAL, "imagenet", "val/imagenet/"),
        pytest.param(Mode.INFER, None, "infer/"),
    ])
    def test_prefix(self, mode, dataset, prefix):
        state = State(mode, dataset)
        assert state.prefix == prefix
