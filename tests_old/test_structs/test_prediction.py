#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Type

import pytest
import torch

from project.structs import BinaryPrediction, MultiClassPrediction, Prediction


class BaseTest:
    CLS: Type[Prediction]

    @pytest.fixture(autouse=True)
    def seed(self):
        torch.random.manual_seed(42)

    @pytest.fixture(params=[torch.float, torch.half])
    def logits(self, request):
        return torch.rand(1, dtype=request.param)

    def test_construct(self, logits):
        p = self.CLS(logits)
        assert p.logits is logits

    def test_repr(self, logits):
        example = self.CLS(logits)
        print(example)

    def test_len(self, logits):
        N = 4
        t = logits.unsqueeze(0).expand(N, -1)
        ex = self.CLS(t)
        assert ex.is_batched
        assert len(ex) == N

    def test_eq(self, logits):
        l1 = logits
        l2 = l1.clone()
        l3 = logits + 0.1

        p1 = self.CLS(l1)
        p2 = self.CLS(l2)
        p3 = self.CLS(l3)

        assert p1 == p2
        assert p1 != p3
        assert p2 != p3

    def test_getitem(self, logits):
        l1 = logits
        l2 = logits + 0.1

        p1 = self.CLS(l1)
        p2 = self.CLS(l2)

        batch = self.CLS.from_unbatched([p1, p2])

        out1 = batch[0]
        out2 = batch[1]
        assert p1 == out1
        assert p2 == out2

    def test_from_unbatched(self, logits):
        l1 = logits
        l2 = logits + 0.1
        p1 = self.CLS(l1)
        p2 = self.CLS(l2)
        batch = self.CLS.from_unbatched([p1, p2])
        assert torch.allclose(batch.logits, torch.stack([l1, l2]))


class TestBinaryPrediction(BaseTest):
    CLS: Type[BinaryPrediction] = BinaryPrediction

    def test_probs(self, logits):
        ex = self.CLS(logits)
        assert torch.allclose(ex.probs, logits.float().sigmoid())

    @pytest.mark.parametrize("threshold", [0.1, 0.5, 0.8])
    def test_classes(self, logits, threshold):
        ex = self.CLS(logits)
        cls = ex.classes(threshold=threshold)
        expected = (logits.float().sigmoid() >= threshold).long()
        assert torch.allclose(cls, expected)

    def test_entropy(self, logits):
        ex = self.CLS(logits)
        entropy = ex.entropy
        assert 0 <= float(entropy.item()) <= 1


class TestMultiClassPrediction(BaseTest):
    CLS: Type[MultiClassPrediction] = MultiClassPrediction

    @pytest.fixture(params=[torch.float, torch.half])
    def logits(self, request):
        return torch.rand(3, dtype=request.param)

    def test_probs(self, logits):
        ex = self.CLS(logits)
        assert torch.allclose(ex.probs, logits.float().softmax(dim=-1))

    def test_classes(self, logits):
        ex = self.CLS(logits)
        cls = ex.classes
        expected = logits.argmax(dim=-1)
        assert torch.allclose(cls, expected)

    def test_entropy(self, logits):
        ex = self.CLS(logits)
        entropy = ex.entropy
        assert 0 <= float(entropy.item()) <= 1

    @pytest.mark.parametrize("batched", [False, True])
    @pytest.mark.parametrize("clazz", [0, 1])
    def test_probs_for_class(self, logits, batched, clazz):
        classes = torch.tensor([clazz])
        if batched:
            logits = logits.unsqueeze(0).expand(4, -1)
            classes = classes.unsqueeze(0).expand(4, -1)

        ex = self.CLS(logits)
        scores = ex.probs_for_class(classes)
        expected = logits.float().softmax(dim=-1)[..., clazz]
        assert torch.allclose(scores, expected)
