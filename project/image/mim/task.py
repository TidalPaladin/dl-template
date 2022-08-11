#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
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


LOSS_FN_REGISTRY( F.mse_loss, name="mse")


class MaskedImageModeling(AdapterTask):
    backbones: Registry = BACKBONE_REGISTRY
    heads: Registry = HEAD_REGISTRY
    adapters: Registry = ADAPTER_REGISTRY

    def __init__(
        self,
        backbone: str = "vit",
        backbone_kwargs: Optional[Dict] = None,
        head: str = "vit-mae",
        pretrained: Union[bool, str] = True,
        loss_fn: Optional[str] = None,
        metrics: METRICS_TYPE = None,
        learning_rate: Optional[float] = None,
        patch_size: Union[int, Tuple[int, int]] = (8, 8),
        channels: int = 3,
        masking_ratio: float = 0.75,
        adapter: str = "mae-default",
        adapter_kwargs: Dict[str, Any] = {},
        output_transform: OUTPUT_TRANSFORM_TYPE = None,
    ):
        self.save_hyperparameters()
        self.masking_ratio = masking_ratio
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

        if not backbone_kwargs:
            backbone_kwargs = {}

        backbone_class = self.backbones.get(backbone)
        num_features = backbone_class.metadata.get("num_features", None)
        backbone_class = backbone_class.bind_metadata(pretrained=pretrained, patch_size=self.patch_size, **backbone_kwargs)
        backbone = backbone_class()

        loss_fn_class = LOSS_FN_REGISTRY.get("mse").fn

        num_features = getattr(backbone, "num_features", None)
        if num_features is None:
            raise ValueError("`num_features` not provided")
        else:
            head_class = self.heads.get(head).bind_metadata(
                num_features=num_features, 
                out_channels=channels, 
                patch_size=self.patch_size,
            )

        adapter_class: Adapter = self.adapters.get(adapter or "default").bind_metadata(
            task=self,
            backbone=backbone,
            head=head_class(),
            pretrained=pretrained,
            **adapter_kwargs,
        )()

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
        task: MaskedImageModeling,
        backbone: nn.Module,
        head: nn.Module,
        **kwargs,
    ) -> Adapter:
        adapter = cls(backbone, head)
        adapter.__dict__["_task"] = task
        return adapter

    @property
    def task(self) -> MaskedImageModeling:
        return cast(MaskedImageModeling, self._task)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        batch = self._prepare_batch(batch)
        return Task.training_step(cast(Task, self.task), batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        batch = self._prepare_batch(batch)
        return Task.validation_step(cast(Task, self.task), batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        batch = self._prepare_batch(batch)
        return Task.test_step(cast(Task, self.task), batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch[DataKeys.PREDS] = Task.predict_step(
            cast(Task, self.task), (batch[DataKeys.INPUT]), batch_idx, dataloader_idx=dataloader_idx
        )
        return batch

    def _prepare_batch(self, batch: Any) -> Tuple[Tensor, Tensor]:
        return (batch[DataKeys.INPUT], batch[DataKeys.INPUT])

    @torch.no_grad()
    def get_mask(self, features: Tensor) -> Tensor:
        assert 0 <= self.task.masking_ratio < 1.0
        N, L, D = features.shape
        mask = (
            torch.rand(L, device=features.device)
            .lt_(self.task.masking_ratio)
            .bool()
        )
        assert mask.device == features.device
        return mask

    def drop_masked_tokens(self, features: Tensor, mask: Tensor) -> Tensor:
        N, Lf, D = features.shape
        with torch.no_grad():
            mask = mask.view(1, -1, 1).expand(N, -1, D)
        features = features[~mask].view(N, -1, D)
        return features

    def forward(self, x) -> Tensor:
        H, W = x.shape[-2:]
        # TODO: Resolve this hack
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # run stem / patch embedder
        x = self.backbone.stem(x)

        # get mask based on stem output tokens
        mask = self.get_mask(x)

        # drop masked tokens and run the remainder through the vit
        x = self.drop_masked_tokens(x, mask)
        x = self.backbone.body(x)

        # run the decoder 
        x = self.head(x, mask, img_size=(H, W))

        return x


ADAPTER_REGISTRY(DefaultAdapter.from_task, name="mae-default")
