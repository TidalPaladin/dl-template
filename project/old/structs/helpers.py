#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import Tuple, TypeVar

from torch import Tensor
import torch


T = TypeVar("T", bound="ResizeMixin")


class ResizeMixin:
    @abstractmethod
    def resize(self: T, scale_factor: float, **kwargs) -> T:
        ...

    @abstractmethod
    def compute_resize_scale(self, max_size: Tuple[int, int]) -> float:
        ...

    def resize_to_fit(self: T, max_size: Tuple[int, int], **kwargs) -> T:
        scale = self.compute_resize_scale(max_size)
        return self.resize(scale, **kwargs) if scale < 1.0 else self

    def _compute_scale(
        self,
        size: Tuple[int, int],
        max_size: Tuple[int, int],
    ) -> float:
        h, w = size
        h_max, w_max = max_size
        ratio_h, ratio_w = h_max / h, w_max / w
        return min(ratio_h, ratio_w)

    @staticmethod
    def _view_for_resize(img: Tensor) -> Tensor:
        size = torch.broadcast_shapes(img.shape, (1, 1, 1, 1))
        return img.view(*size)

    @staticmethod
    def _view_as_orig(img: Tensor, orig: Tensor) -> Tensor:
        H, W = img.shape[-2:]
        if orig.ndim == 2:
            return img.view(H, W)
        elif orig.ndim == 3:
            return img.view(-1, H, W)
        elif orig.ndim == 4:
            B = orig.shape[0]
            return img.view(B, -1, H, W)
        else:
            raise ValueError(f"Unexpected image shape {img.shape}")
