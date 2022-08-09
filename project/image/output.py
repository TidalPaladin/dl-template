#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Optional

import torch
from flash.core.data.io.input import DataKeys
from flash.core.data.io.output import Output
from flash.core.data.io.output_transform import OutputTransform
from flash.core.model import OutputKeys
from matplotlib.cm import get_cmap
from PIL import Image
from registry.registry import Registry
from torch import Tensor


OUTPUT_REGISTRY = Registry("outputs", bound=Output)
OUTPUT_REGISTRY(name="raw")(Output)


class MyOutputTransform(OutputTransform):
    def per_sample_transform(self, sample: Any) -> Any:
        pred: Tensor = sample[OutputKeys.OUTPUT]
        C, L = pred.shape
        shape = torch.Size(sample[DataKeys.METADATA]["shape"])
        pred = pred.clamp(min=0, max=1)
        sample[OutputKeys.OUTPUT] = pred.view(C, *shape)
        sample[OutputKeys.TARGET] = sample[OutputKeys.TARGET].view(C, *shape)
        return sample


@OUTPUT_REGISTRY(name="image")
class ImageOutput(Output):
    def __init__(self, include_target: bool = True, cmap: Optional[str] = None):
        super().__init__()
        self.include_target = include_target
        self.cmap = cmap

    def transform(self, sample: Dict[str, Any]) -> Image.Image:
        pred: Tensor = sample[OutputKeys.OUTPUT]
        target = sample.get(OutputKeys.TARGET, None)
        if self.include_target and target is not None:
            pred = torch.cat([pred, target], dim=-1)
        img = self.to_image(pred)
        return img

    def to_image(self, t: Tensor) -> Image.Image:
        if t.ndim > 3:
            t = t.amax(dim=1)
        assert t.ndim == 3, f"invalid shape {t.shape}"
        t = t.cpu()
        # if self.cmap is not None:
        #    assert t.shape[0] == 1
        #    t = self.apply_colormap(t)
        pixels = t.mul(255).movedim(0, -1).squeeze_().byte().numpy()
        img = Image.fromarray(pixels)
        return img

    def apply_colormap(self, x: Tensor) -> Tensor:
        assert self.cmap is not None
        cm_hot = get_cmap(self.cmap)
        torch.from_numpy(cm_hot(x))
        return x
