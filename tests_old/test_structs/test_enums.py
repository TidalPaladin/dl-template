#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from project.structs import Mode, State


class TestMode:
    @pytest.mark.parametrize(
        "mode,prefix",
        [
            pytest.param(Mode.TRAIN, "train"),
            pytest.param(Mode.VAL, "val"),
            pytest.param(Mode.TEST, "test"),
            pytest.param(Mode.INFER, "infer"),
        ],
    )
    def test_prefix(self, mode, prefix):
        assert mode.prefix == prefix

    @pytest.mark.parametrize(
        "mode,string",
        [
            pytest.param(Mode.TRAIN, "train"),
            pytest.param(Mode.VAL, "val"),
            pytest.param(Mode.TEST, "test"),
            pytest.param(Mode.INFER, "infer"),
            pytest.param(None, "foo", marks=pytest.mark.xfail(raises=ValueError)),
        ],
    )
    def test_from_str(self, mode, string):
        assert Mode.from_str(string) == mode

    @pytest.mark.parametrize(
        "group,expected",
        [
            pytest.param(["train", "test"], [Mode.TRAIN, Mode.TEST]),
            pytest.param(["train", Mode.TEST], [Mode.TRAIN, Mode.TEST]),
            pytest.param([Mode.VAL], [Mode.VAL]),
            pytest.param([], []),
            pytest.param([42], [], marks=pytest.mark.xfail(raises=TypeError)),
        ],
    )
    def test_from_group(self, group, expected):
        assert Mode.from_group(group) == expected


class TestState:
    @pytest.mark.parametrize(
        "mode,dataset",
        [
            pytest.param(Mode.TRAIN, "train"),
            pytest.param(Mode.TEST, "cifar10"),
            pytest.param(Mode.VAL, "imagenet"),
            pytest.param(Mode.INFER, None),
        ],
    )
    def test_construct(self, mode, dataset):
        state = State(mode, dataset)
        assert state.mode == mode
        assert state.dataset == dataset

    @pytest.mark.parametrize(
        "mode,dataset",
        [
            pytest.param(Mode.TRAIN, "train"),
            pytest.param(Mode.TEST, "cifar10"),
            pytest.param(Mode.VAL, "imagenet"),
            pytest.param(Mode.INFER, None),
        ],
    )
    def test_update(self, mode, dataset):
        state = State(Mode.INFER, None)
        new_state = state.update(mode, dataset)
        assert new_state.mode == mode
        assert new_state.dataset == dataset

    @pytest.mark.parametrize(
        "mode,dataset,prefix",
        [
            pytest.param(Mode.TRAIN, "train", "train/"),
            pytest.param(Mode.TRAIN, None, "train/"),
            pytest.param(Mode.TEST, "cifar10", "test/cifar10/"),
            pytest.param(Mode.VAL, "imagenet", "val/imagenet/"),
            pytest.param(Mode.INFER, None, "infer/"),
        ],
    )
    def test_prefix(self, mode, dataset, prefix):
        state = State(mode, dataset)
        assert state.prefix == prefix

    @pytest.mark.parametrize(
        "state1,state2,eq",
        [
            pytest.param(State(Mode.TRAIN, None), State(Mode.TRAIN, None), True),
            pytest.param(State(Mode.TRAIN, None), State(Mode.TRAIN, "foo"), False),
            pytest.param(State(Mode.TEST, None), State(Mode.TRAIN, None), False),
            pytest.param(State(Mode.TRAIN, None, True), State(Mode.TRAIN, None, False), True),
        ],
    )
    def test_eq(self, state1, state2, eq):
        assert (state1 == state2) == eq

    @pytest.mark.parametrize(
        "state1,state2,eq",
        [
            pytest.param(State(Mode.TRAIN, None), State(Mode.TRAIN, None), True),
            pytest.param(State(Mode.TRAIN, None), State(Mode.TRAIN, "foo"), False),
            pytest.param(State(Mode.TEST, None), State(Mode.TRAIN, None), False),
            pytest.param(State(Mode.TRAIN, None, True), State(Mode.TRAIN, None, False), True),
        ],
    )
    def test_hash_eq(self, state1, state2, eq):
        assert (hash(state1) == hash(state2)) == eq
