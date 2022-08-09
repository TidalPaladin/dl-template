#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash.core.adapter import Adapter, AdapterTask
from flash.core.classification import ClassificationMixin
from flash.core.data.io.input import DataKeys
from flash.core.utilities.types import METRICS_TYPE, OUTPUT_TRANSFORM_TYPE
from flash.core.utilities.url_error import catch_url_error
from registry import Registry
from torch import Tensor
from torchmetrics import Accuracy, F1Score, Metric

from ...core.adapter import ADAPTER_REGISTRY
from ...core.task import BACKBONE_REGISTRY, HEAD_REGISTRY, LOSS_FN_REGISTRY, Task


class ClassificationMixin:
    def _build(
        self,
        num_classes: Optional[int] = None,
        labels: Optional[List[str]] = None,
        loss_fn: Optional[Callable] = None,
        metrics: Union[Metric, Mapping, Sequence, None] = None,
        multi_label: bool = False,
    ):
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.labels = labels

        if metrics is None:
            metrics = F1Score(num_classes) if (multi_label and num_classes) else Accuracy()

        # if loss_fn is None:
        #    loss_fn = F.cross_entropy if multi_label else F.binary_cross_entropy_with_logits

        return metrics, loss_fn

    def to_metrics_format(self, x: Tensor) -> Tensor:
        if getattr(self, "multi_label", False):
            return torch.sigmoid(x)
        return torch.softmax(x, dim=1)


@LOSS_FN_REGISTRY(name="bce")
def bce_loss(x: Tensor, y: Tensor) -> Tensor:
    """Calls BCE with logits and cast the target one_hot (y) encoding to floating point precision."""
    return F.binary_cross_entropy_with_logits(x, y.float())


class ImageClassifier(AdapterTask, ClassificationMixin):
    backbones: Registry = BACKBONE_REGISTRY
    heads: Registry = HEAD_REGISTRY
    adapters: Registry = ADAPTER_REGISTRY

    def __init__(
        self,
        num_classes: Optional[int] = None,
        labels: Optional[List[str]] = None,
        backbone: str = "basic-conv",
        backbone_kwargs: Optional[Dict] = None,
        head: str = "linear",
        pretrained: Union[bool, str] = True,
        loss_fn: Optional[str] = None,
        metrics: METRICS_TYPE = None,
        learning_rate: Optional[float] = None,
        multi_label: bool = False,
        adapter: str = "default",
        adapter_kwargs: Dict[str, Any] = {},
        output_transform: OUTPUT_TRANSFORM_TYPE = None,
    ):
        self.save_hyperparameters()

        if labels is not None and num_classes is None:
            num_classes = len(labels)

        if not backbone_kwargs:
            backbone_kwargs = {}

        backbone_class = self.backbones.get(backbone)
        num_features = backbone_class.metadata.get("num_features", None)
        backbone_class = backbone_class.bind_metadata(pretrained=pretrained, **backbone_kwargs)
        backbone = backbone_class()

        loss_fn_class = LOSS_FN_REGISTRY.get("bce").fn

        num_features = getattr(backbone, "num_features", None)
        if num_features is None:
            raise ValueError("`num_features` not provided")
        else:
            head_class = self.heads.get(head).bind_metadata(num_features=num_features, num_classes=num_classes)

        adapter_class: Adapter = self.adapters.get(adapter or "default").bind_metadata(
            task=self,
            num_classes=num_classes,
            backbone=backbone,
            head=head_class(),
            pretrained=pretrained,
            **adapter_kwargs,
        )()
        metrics, loss_fn = self._build(num_classes, labels, loss_fn, metrics, multi_label)

        super().__init__(
            adapter_class,
            loss_fn=loss_fn_class,
            metrics=metrics,
            learning_rate=learning_rate,
        )


class DefaultAdapter(Adapter):
    """The ``DefaultAdapter`` is an :class:`~flash.core.adapter.Adapter`."""

    required_extras: str = "image"

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()

        self.backbone = backbone
        self.head = head

    @classmethod
    @catch_url_error
    def from_task(
        cls,
        *args,
        task: AdapterTask,
        backbone: nn.Module,
        head: nn.Module,
        **kwargs,
    ) -> Adapter:
        adapter = cls(backbone, head)
        adapter.__dict__["_task"] = task
        return adapter

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return Task.training_step(self._task, batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return Task.validation_step(self._task, batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return Task.test_step(self._task, batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch[DataKeys.PREDS] = Task.predict_step(
            self._task, (batch[DataKeys.INPUT]), batch_idx, dataloader_idx=dataloader_idx
        )
        return batch

    def forward(self, x) -> Tensor:
        # TODO: Resolve this hack
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.backbone(x)
        if x.dim() == 4:
            x = x.mean(-1).mean(-1)
        return self.head(x)


ADAPTER_REGISTRY(DefaultAdapter.from_task, name="default")
