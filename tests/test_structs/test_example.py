#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pytest
import torch

from project.structs import Example


class TestExample:
    @pytest.fixture(autouse=True)
    def seed(self):
        torch.random.manual_seed(42)

    def test_construct(self):
        img = torch.rand(3, 32, 32)
        label = torch.tensor(0).view(1)
        example = Example(img, label)
        assert example.img is img
        assert example.label is label

    def test_repr(self):
        img = torch.rand(3, 32, 32)
        example = Example(img)
        print(example)

    @pytest.mark.parametrize("label", [True, False])
    def test_has_label(self, label):
        img = torch.rand(3, 32, 32)
        l = torch.rand(1) if label else None
        example = Example(img, l)
        assert example.has_label == label

    def test_len(self):
        N = 4
        t = torch.rand(N, 3, 32, 32)
        ex = Example(t, None)
        assert len(ex) == N

    @pytest.mark.parametrize("label", [True, False])
    def test_eq(self, label):
        t1 = torch.rand(3, 32, 32)
        t2 = t1.clone()
        t3 = torch.rand(3, 32, 32)
        if label:
            l1 = torch.tensor([0])
            l2 = l1.clone()
            l3 = torch.tensor([1])
        else:
            l1 = None
            l2 = None
            l3 = None

        e1 = Example(t1, l1)
        e2 = Example(t2, l2)
        e3 = Example(t3, l3)

        assert e1 == e2
        assert e1 != e3
        assert e2 != e3

    @pytest.mark.parametrize("label", [True, False])
    def test_getitem(self, label):
        t1 = torch.rand(3, 32, 32)
        t2 = torch.rand(3, 32, 32)
        if label:
            l1 = torch.tensor([0])
            l2 = torch.tensor([1])
        else:
            l1 = None
            l2 = None
        e1 = Example(t1, l1)
        e2 = Example(t2, l2)
        batch = Example.from_unbatched([e1, e2])

        out1 = batch[0]
        out2 = batch[1]

        assert e1 == out1
        assert e2 == out2

    @pytest.mark.parametrize("label", [True, False])
    def test_from_unbatched(self, label):
        t1 = torch.rand(3, 32, 32)
        t2 = torch.rand(3, 32, 32)
        if label:
            l1 = torch.tensor([0])
            l2 = torch.tensor([1])
        else:
            l1 = None
            l2 = None
        e1 = Example(t1, l1)
        e2 = Example(t2, l2)
        batch = Example.from_unbatched([e1, e2])

        assert torch.allclose(batch.img, torch.stack([t1, t2]))
        if label:
            assert torch.allclose(batch.label, torch.stack([l1, l2]))  # type: ignore
        else:
            assert batch.label is None

    @pytest.mark.parametrize(
        "shape",
        [
            (128, 128),
            (3, 128, 128),
            (1, 128, 128),
            (4, 3, 128, 128),
        ],
    )
    @pytest.mark.parametrize("scale", [0.25, 0.5])
    def test_resize(self, shape, scale):
        img = torch.rand(*shape)
        H, W = shape[-2:]
        example = Example(img)

        out = example.resize(scale_factor=scale)
        H_out, W_out = out.img.shape[-2:]
        assert H_out == int(H * scale)
        assert W_out == int(W * scale)

    @pytest.mark.parametrize(
        "shape",
        [
            (128, 128),
            (3, 128, 128),
            (1, 128, 128),
            (4, 3, 128, 128),
            (4, 3, 128, 64),
        ],
    )
    @pytest.mark.parametrize(
        "max_size",
        [
            (128, 128),
            (64, 64),
            (64, 32),
        ],
    )
    def test_resize_to_fit(self, shape, max_size):
        img = torch.rand(*shape)
        H, W = shape[-2:]
        example = Example(img)

        out = example.resize_to_fit(max_size)
        H_out, W_out = out.img.shape[-2:]
        assert H // W == H_out // W_out, "aspect ratio should be preserved"
        assert H_out <= max_size[0]
        assert W_out <= max_size[1]

    def test_require_grad(self):
        img = torch.rand(3, 32, 32)
        label = torch.tensor(0).view(1)
        example = Example(img, label)
        copy = example.require_grad()
        assert example.img.requires_grad
        assert copy is example
