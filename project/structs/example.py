#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Type, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor

from combustion.util import MISSING
from combustion.util.dataclasses import BatchMixin, TensorDataclass

from .helpers import ResizeMixin


I = TypeVar("I", bound="Example")


@dataclass(repr=False)
class Example(TensorDataclass, BatchMixin, ResizeMixin):
    r"""Base container for training examples."""
    img: Tensor
    label: Optional[Tensor] = None

    __slice_fields__ = ["img", "label"]

    def __post_init__(self):
        if not 2 <= self.img.ndim <= 4:
            raise ValueError(f"Unexpected shape {self.img.shape} for `img`")
        if self.label is not None:
            if not self.label.ndim:
                self.label.unsqueeze_(0)
            elif not (1 <= self.label.ndim <= 2):
                raise ValueError(f"Unexpected shape {self.label.shape} for `label`")

    def __eq__(self, other: "Example") -> bool:
        img_match = torch.allclose(self.img, other.img)
        if self.has_label and other.has_label:
            label_match = torch.allclose(self.label, other.label)  # type: ignore
        elif not self.has_label and not other.has_label:
            label_match = True
        else:
            label_match = False
        return img_match and label_match

    def require_grad(self: I) -> I:
        self.img.requires_grad = True
        return self

    @property
    def is_batched(self) -> bool:
        return self.img.ndim == 4

    @classmethod
    def from_unbatched(cls: Type[I], examples: Iterable[I]) -> I:
        assert not any(ex.is_batched for ex in examples)
        img = torch.stack([ex.img for ex in examples], dim=0)

        has_label = set(ex.has_label for ex in examples)
        assert len(has_label) == 1, "Cannot combine examples with/without labels"

        if all(ex.has_label for ex in examples):
            label = torch.stack([ex.label for ex in examples], dim=0)  # type: ignore
        elif not any(ex.has_label for ex in examples):
            label = None
        else:
            raise RuntimeError("Cannot combine examples with/without labels")

        return cls(img, label)

    def __len__(self) -> int:
        if not self.is_batched:
            return 0
        assert self.label is None or self.label.shape[0] == self.img.shape[0]
        return self.img.shape[0]

    @property
    def has_label(self) -> bool:
        return self.label not in (None, MISSING)

    @classmethod
    def create(cls: Type[I], *args, **kwargs) -> I:
        r"""Creation method that supports passthrough when the only arg is
        an :class:`Example` instance.
        """
        if len(args) == 1 and not kwargs and isinstance(args[0], cls):
            return args[0]
        return cls(*args, **kwargs)

    def resize(
        self: I,
        scale_factor: float,
        mode: str = "bilinear",
        align_corners=None,
        recompute_scale_factor: bool = False,
        **kwargs,
    ) -> I:
        orig_img = self.img
        img = self._view_for_resize(orig_img)
        img = F.interpolate(
            img,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,  # type: ignore
            **kwargs,
        )
        img = self._view_as_orig(img, orig_img)
        assert img.ndim == orig_img.ndim
        return self.replace(img=img)

    def compute_resize_scale(self, max_size: Tuple[int, int]) -> float:
        H, W = self.img.shape[-2:]
        return self._compute_scale((H, W), max_size)
