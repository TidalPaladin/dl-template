#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import pytest
from project.metrics import StateCollection, MetricStateCollection, QueueStateCollection
from project.structs import State, Mode
from torchmetrics import MetricCollection, Accuracy, F1
from typing import Type, Any
from queue import PriorityQueue

class SimpleMetricCollection(MetricStateCollection):

    def __init__(self):
        collection = MetricCollection({
            f"accuracy": Accuracy(),
            f"f1": F1(),
        })
        super().__init__(collection)


class BaseCollectionTest:
    CLS: Type[StateCollection]
    VAL: Any

    simple_states =  [
        pytest.param(State(Mode.TRAIN, None)),
        pytest.param(State(Mode.VAL, "cifar10")),
        pytest.param(State(Mode.TEST, "imagenet")),
    ]
    
    @pytest.mark.parametrize("state", simple_states)
    def test_register(self, state):
        col = self.CLS()
        col.register(state)
        assert state in col.states

    def test_hash(self):
        col = self.CLS()
        col2 = self.CLS()
        assert hash(col) != hash(col2)

    @pytest.mark.parametrize("state", simple_states)
    def test_set_state(self, state):
        col = self.CLS()
        col.register(state)
        col.set_state(state, self.VAL)
        assert state in col.states

    @pytest.mark.parametrize("state", simple_states)
    def test_get_state(self, state):
        col = self.CLS()
        col.register(state)
        get = col.get_state(state)
        assert isinstance(get, type(self.VAL))

    @pytest.mark.parametrize("state", simple_states)
    def test_remove_state(self, state):
        col = self.CLS()
        col.register(state)
        assert state in col.states
        col.remove_state(state)
        assert state not in col.states

    @pytest.mark.parametrize("state", simple_states)
    def test_as_dict(self, state):
        col = self.CLS()
        col.register(state)
        d = col.as_dict()
        assert isinstance(d, dict)
        assert len(d) == 1
        assert state in d.keys()

    @pytest.mark.parametrize("state", simple_states)
    def test_add(self, state):
        col = self.CLS()
        col.register(state)

        other_col = self.CLS()
        other_state = State(Mode.INFER, "other_state")
        other_col.register(other_state)

        added = col + col + other_col
        assert isinstance(other_col, self.CLS)
        assert added.states == {state, other_state}


class TestMetricStateCollection(BaseCollectionTest):
    CLS: Type[MetricStateCollection] = SimpleMetricCollection
    VAL = MetricCollection({})

    def test_to_cpu(self):
        col = self.CLS()
        state = State(Mode.TEST, "cifar10")
        col.register(state)
        device = torch.device("cpu")
        col2 = col.to(device)
        for s, collection in col2.as_dict().items():
            for name, metric in collection.items():
                assert metric.device == device

    @pytest.mark.cuda_or_skip
    def test_to_gpu(self):
        col = self.CLS()
        state = State(Mode.TEST, "cifar10")
        state2 = State(Mode.TRAIN, "cifar10")
        col.register(state)
        col.register(state2)
        device = torch.device("cuda:0")
        col2 = col.to(device)
        for s, collection in col2.as_dict().items():
            for name, metric in collection.items():
                assert metric.device == device

    def test_reset(self):
        col = self.CLS()
        state = State(Mode.TEST, "cifar10")
        state2 = State(Mode.TRAIN, "cifar10")
        col.register(state)
        col.register(state2)

        p = torch.rand(10, 10)
        t = torch.rand(10, 10).round().long()

        col.get_state(state).update(p, t)
        col.get_state(state2).update(p, t)
        col2 = col.reset()

        for s, collection in col2.as_dict().items():
            for name, metric in collection.items():
                assert metric.tp.item() == 0 #type: ignore
                assert metric.fp.item() == 0 #type: ignore
                assert metric.tn.item() == 0 #type: ignore
                assert metric.fn.item() == 0 #type: ignore


class TestQueueStateCollection(BaseCollectionTest):
    CLS: Type[QueueStateCollection] = QueueStateCollection
    VAL = PriorityQueue()

    @pytest.mark.parametrize("state", BaseCollectionTest.simple_states)
    def test_enqueue(self, state):
        col = self.CLS()
        col.register(state)
        col.enqueue(state, 0, "dog") 
        col.enqueue(state, 1, "cat") 
        queue = col.get_state(state)
        assert queue.qsize() == col.qsize(state) == 2
        assert not col.empty(state)

    @pytest.mark.parametrize("state", BaseCollectionTest.simple_states)
    def test_dequeue(self, state):
        col = self.CLS()
        col.register(state)
        col.enqueue(state, 1, "cat") 
        col.enqueue(state, 0, "dog") 
        queue = col.get_state(state)
        assert queue.qsize() == 2

        item1 = col.dequeue(state)
        item2 = col.dequeue(state)
        assert item1.priority <= item2.priority
        assert item1.item == "dog"
        assert item2.item == "cat"

    @pytest.mark.parametrize("state", BaseCollectionTest.simple_states)
    def test_len(self, state):
        col = self.CLS()
        col.register(state)
        state2 = State(Mode.INFER, "state2")
        col.register(state2)

        col.enqueue(state, 0, "dog") 
        col.enqueue(state, 1, "cat") 
        assert len(col) == 2
        col.enqueue(state2, 0, "dog") 
        col.enqueue(state2, 1, "cat") 
        assert len(col) == 4

    def test_reset(self):
        col = self.CLS()
        state = State(Mode.TEST, "cifar10")
        state2 = State(Mode.TRAIN, "cifar10")
        col.register(state)
        col.register(state2)

        p = torch.rand(10, 10)
        col.enqueue(state, 0, p)
        col.enqueue(state2, 0, p)
        col2 = col.reset()

        for s, queue in col2.as_dict().items():
            assert queue.empty()
