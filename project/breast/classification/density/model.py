#!/usr/bin/env python
# -*- coding: utf-8 -*-

from project.core.task import Task


from types import FunctionType
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn

from flash.core.classification import ClassificationAdapterTask, ClassificationMixin
from flash.core.data.io.input import ServeInput
from flash.core.data.io.output import Output
from flash.core.registry import FlashRegistry
from flash.core.serve import Composition
from flash.core.utilities.imports import requires
from flash.core.utilities.types import (
    INPUT_TRANSFORM_TYPE,
    LOSS_FN_TYPE,
    LR_SCHEDULER_TYPE,
    METRICS_TYPE,
    OPTIMIZER_TYPE,
)
from flash.image.classification.adapters import TRAINING_STRATEGIES
from flash.image.classification.backbones import IMAGE_CLASSIFIER_BACKBONES
from flash.image.classification.heads import IMAGE_CLASSIFIER_HEADS
from flash.image.classification.input_transform import ImageClassificationInputTransform
from flash.image.data import ImageDeserializer



class BreastDensityTask(ClassificationMixin, Task):
    backbones: FlashRegistry = IMAGE_CLASSIFIER_BACKBONES
    heads: FlashRegistry = IMAGE_CLASSIFIER_HEADS
    required_extras: str = "image"

    def __init__(
        self,
        num_classes: Optional[int] = 4,
        labels: Optional[List[str]] = ["A", "B", "C", "D"],
        backbone: Union[str, Tuple[nn.Module, int]] = "convnext_base",
        backbone_kwargs: Optional[Dict] = None,
        head: Union[str, FunctionType, nn.Module] = "linear",
        pretrained: Union[bool, str] = True,
        loss_fn: LOSS_FN_TYPE = None,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: METRICS_TYPE = None,
        learning_rate: float = 1e-3,
        multi_label: bool = False,
    ):
        self.save_hyperparameters()

        if labels is not None and num_classes is None:
            num_classes = len(labels)

        if not backbone_kwargs:
            backbone_kwargs = {}


        if isinstance(backbone, tuple):
            backbone, num_features = backbone
        else:
            backbone, num_features = self.backbones.get(backbone)(pretrained=pretrained, **backbone_kwargs)

        if isinstance(head, str):
            head = self.heads.get(head)(num_features=num_features, num_classes=num_classes)
        else:
            head = head(num_features, num_classes) if isinstance(head, FunctionType) else head

        adapter_from_class = self.training_strategies.get("default")
        adapter = adapter_from_class(
            task=self,
            num_classes=num_classes,
            backbone=backbone,
            head=head,
            pretrained=pretrained,
        )

        super().__init__(
            num_classes=num_classes,
            loss_fn=loss_fn,
            metrics=metrics,
            learning_rate=learning_rate,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            labels=labels,
        )

    @classmethod
    def available_pretrained_weights(cls, backbone: str):
        result = cls.backbones.get(backbone, with_metadata=True)
        pretrained_weights = None

        if "weights_paths" in result["metadata"]:
            pretrained_weights = list(result["metadata"]["weights_paths"].keys())

        return pretrained_weights
