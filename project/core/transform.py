#!/usr/bin/env python
# -*- coding: utf-8 -*-
from icevision.tfms import A

from flash import InputTransform
from flash.core.integrations.icevision.transforms import IceVisionTransformAdapter
from flash.image import ObjectDetectionData


from dataclasses import dataclass
from typing import Callable, Tuple, Union

import torch

from flash.core.data.io.input_transform import InputTransform
from flash.core.data.transforms import kornia_collate
from flash.core.utilities.imports import _ALBUMENTATIONS_AVAILABLE, _TORCHVISION_AVAILABLE, requires

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as T

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as alb


class AlbumentationsAdapter(torch.nn.Module):
    @requires("albumentations")
    def __init__(self, transform):
        super().__init__()
        if not isinstance(transform, list):
            transform = [transform]
        self.transform = alb.Compose(transform)

    def forward(self, x):
        return torch.from_numpy(self.transform(image=x.numpy())["image"])


@dataclass
class ImageClassificationInputTransform(InputTransform):

    image_size: Tuple[int, int] = (196, 196)
    mean: Union[float, Tuple[float, float, float]] = (0.485, 0.456, 0.406)
    std: Union[float, Tuple[float, float, float]] = (0.229, 0.224, 0.225)

    def input_per_sample_transform(self):
        return T.Compose([T.ToTensor(), T.Resize(self.image_size), T.Normalize(self.mean, self.std)])

    def train_input_per_sample_transform(self):
        return T.Compose(
            [T.ToTensor(), T.Resize(self.image_size), T.Normalize(self.mean, self.std), T.RandomHorizontalFlip()]
        )

    def target_per_sample_transform(self) -> Callable:
        return torch.as_tensor

    def collate(self) -> Callable:
        # TODO: Remove kornia collate for default_collate
        return kornia_collate
