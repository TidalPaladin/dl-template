#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as xform

from torch import Tensor
from dataclasses import dataclass
from ..structs import Example, Loss, MultiClassPrediction, Prediction
from .base import BaseModel
from combustion.nn.modules.transformer import MLP, LearnableFourierFeatures
from combustion.util.dataclasses import TensorDataclass, BatchMixin
from typing import Any, Tuple, Optional, cast, List, Iterable


@dataclass(repr=False)
class ViTPrediction(MultiClassPrediction):
    features: List[Tensor]
    masked_features: Optional[List[Tensor]] = None
    masked_pred: Optional[Tensor] = None
    __slice_fields__ = MultiClassPrediction.__slice_fields__ + ["features", "masked_features", "masked_pred"]


@dataclass(repr=False)
class SSLLoss(Loss):
    ssl_loss: Tensor

    @property
    def total_loss(self) -> Tensor:
        return self.cls_loss * 0.01 + self.ssl_loss


@dataclass(repr=False)
class MaskIndices(TensorDataclass, BatchMixin):
    h_min: Tensor
    h_max: Tensor
    w_min: Tensor
    w_max: Tensor

    __slice_fields__ = ["h_min", "h_max", "w_min", "w_max"]

    @property
    def is_batched(self) -> bool:
        return self.h_min.numel() > 1

    def __len__(self) -> int:
        if not self.is_batched:
            return 0
        return self.h_min.numel()

    @classmethod
    def from_unbatched(cls, indices: Iterable["MaskIndices"]) -> "MaskIndices":
        return cls(
            torch.stack([idx.h_min for idx in indices]).view(-1),
            torch.stack([idx.h_max for idx in indices]).view(-1),
            torch.stack([idx.w_min for idx in indices]).view(-1),
            torch.stack([idx.w_max for idx in indices]).view(-1),
        )

    @torch.no_grad()
    def _to_mask(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            N, _, H, W = x.shape
        else:
            _, H, W = x.shape
            N = 1
        assert self.h_min.numel() == N, f"{N}, {self.h_min}"

        grid = torch.stack(
            torch.meshgrid(
                torch.arange(H, device=x.device), 
                torch.arange(W, device=x.device), 
                indexing="ij"
            )
        )
        grid = grid.unsqueeze_(0).expand(N, -1, -1, -1)

        # relative to absolute
        h_min = (self.h_min * H).round_().long()
        h_max = (self.h_max * H).round_().long()
        w_min = (self.w_min * H).round_().long()
        w_max = (self.w_max * H).round_().long()

        mask = (
            (grid[..., 0, :, :] >= h_min.view(-1, 1, 1))
            &
            (grid[..., 0, :, :] <= h_max.view(-1, 1, 1))
            &
            (grid[..., 1, :, :] >= w_min.view(-1, 1, 1))
            &
            (grid[..., 1, :, :] <= w_max.view(-1, 1, 1))
        ).unsqueeze_(1)
        assert mask.shape == (N, 1, H, W)
        if x.ndim == 3:
            mask.squeeze_(0)
        assert mask.ndim == x.ndim
        return mask

    def select(self, x: Tensor) -> Tensor:
        mask = self._to_mask(x)
        return x[mask.expand_as(x)] 

    def fill(self, x: Tensor, fill_value: Any = 0) -> Tensor:
        mask = self._to_mask(x)
        x[mask.expand_as(x)] = fill_value
        return x

    @classmethod
    def create(cls, min: float, max: float, N: int) -> "MaskIndices":
        assert 0 < min <= max < 1
        delta = max - min

        h = torch.rand(N).mul_(delta).add_(min)
        w = torch.rand(N).mul_(delta).add_(min)

        low_h = torch.min(torch.rand(N), 1-h)
        low_w = torch.min(torch.rand(N), 1-w)
        high_h = low_h + h
        high_w = low_w + w
        return cls(low_h, high_h, low_w, high_w)


@dataclass
class ViTExample(Example):
    mask_indices: Optional[MaskIndices] = None
    masked_img: Optional[Tensor] = None
    __slice_fields__ = Example.__slice_fields__ + ["mask_indices", "masked_img"]

    def create_mask_indices(self, min: int = 1, max: int = 8) -> MaskIndices:
        N, _, H, W = self.img.shape
        mask_indices = MaskIndices.create(min, max, N)
        return mask_indices

    @classmethod
    def from_example(cls, example: Example) -> "ViTExample":
        return cls(example.img, example.label)


class ViTModel(BaseModel[ViTExample, Prediction, Loss]):
    r"""Basic convolutional model"""
    example_input_array = Example(img=torch.rand(1, 3, 32, 32))
    example_type = ViTExample

    def __init__(
        self,
        width: int = 512,
        num_layers: int = 6,
        ssl_layers: int = 3,
        num_classes: int = 10,
        optimizer_init: dict = {},
        lr_scheduler_init: dict = {},
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_monitor: str = "train/total_loss_epoch",
    ):
        super().__init__(
            num_classes,
            optimizer_init,
            lr_scheduler_init,
            lr_scheduler_interval,
            lr_scheduler_monitor,
        )
        self.save_hyperparameters()
        self.ssl_layers = ssl_layers

        D = width

        self.patch = nn.Conv2d(3, D, 2, stride=2)

        self.body = nn.ModuleList([
            nn.TransformerEncoderLayer(D, nhead=4, dim_feedforward=D*2, dropout=0.1)
            for _ in range(num_layers)
        ])
            
        self.pos_enc = LearnableFourierFeatures(2, 32, D, dropout=0)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Conv2d(D, self.num_classes, 1),
        )
        self.ssl_head = nn.Sequential(
            nn.Conv2d(D, 3*4, 1),
            nn.PixelShuffle(2)
        )

    def forward_features(self, img: Tensor) -> List[Tensor]:
        x = self.patch(img)

        N, _, H, W = x.shape
        L = H*W
        x = x.view(N, -1, L).movedim(-1, 0)

        pos = self.pos_enc.from_grid(dims=(H, W), proto=x, batch_size=N)
        x = x + pos

        features: List[Tensor] = []
        for layer in self.body:
            x = layer(x)
            features.append(x.movedim(0, -1).view(N, -1, H, W))

        return features

    def forward(self, example: ViTExample) -> Prediction:
        img = example.img
        N, C, H, W = img.shape
        features = self.forward_features(img)
        logits = self.head(features[-1]).view(N, -1)

        masked_img = example.masked_img
        assert masked_img is not None
        masked_features = self.forward_features(masked_img)[-(self.ssl_layers):]
        features = features[-(self.ssl_layers):]
        masked_pred = self.ssl_head(masked_features[-1])

        pred = ViTPrediction(logits, features, masked_features, masked_pred)
        return pred

    def compute_loss(self, example: ViTExample, pred: ViTPrediction) -> SSLLoss:
        assert example.label is not None
        example.label
        cls_loss = F.cross_entropy(pred.logits, example.label)
        ssl_loss = torch.zeros_like(cls_loss)
        if pred.masked_features is not None:
            for pf, tf in zip(pred.masked_features, pred.features):
                pf = example.mask_indices.select(pf)
                tf = example.mask_indices.select(tf.detach())
                with torch.no_grad():
                    keep = (pf != 0).logical_and_(tf != 0)
                pf = pf[keep]
                tf = tf[keep]

                assert pred.masked_pred is not None
                pi = example.mask_indices.select(pred.masked_pred)
                ti = example.mask_indices.select(example.img)

                #ssl_loss = ssl_loss + F.mse_loss(pf, tf) + F.mse_loss(pi, ti)
                ssl_loss = ssl_loss + F.mse_loss(pi, ti)

        loss = SSLLoss(cls_loss, ssl_loss)
        return loss

    @torch.no_grad()
    def log_loss(self, loss: SSLLoss) -> None:
        super().log_loss(loss)
        for attr in ("cls_loss", "ssl_loss"):
            name = self.state.with_postfix(attr)
            self._log_loss(name, getattr(loss, attr))

    def step(self, example: ViTExample, batch_idx: int, *args) -> Tuple[Prediction, Loss]:
        # forward pass
        pred: ViTPrediction = self(example)
        assert isinstance(pred, Prediction), "model should return a Prediction instance"

        # compute loss
        loss = self.compute_loss(example, pred)
        assert isinstance(loss, Loss), "compute_loss should return a Loss instance"

        # update metrics
        self.state_metrics.update(self.state, example, pred)

        # loss and metric logging
        self.log_loss(loss)
        self.state_metrics.log(self.state, self)

        return pred, loss

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int):
        r"""Wraps the input batch in an ``Example`` instance if necessary"""
        transform = xform.AutoAugment(xform.AutoAugmentPolicy.CIFAR10)
        batch = super().on_before_batch_transfer(batch, dataloader_idx)

        if self.training:
            batch.img = transform(batch.img.mul(255).byte()).float().div(255)
            batch.img.requires_grad = True

        batch = ViTExample.from_example(batch)
        batch.mask_indices = batch.create_mask_indices(min=0.1, max=0.25)

        img = batch.img.clone().detach()
        img = batch.mask_indices.fill(img, fill_value=0)
        img.requires_grad = True
        batch.masked_img = img

        return batch

class SmallViTModel(ViTModel):
    r"""Basic convolutional model"""

    def __init__(
        self,
        width: int = 32,
        num_layers: int = 3,
        ssl_layers: int = 1,
        num_classes: int = 10,
        optimizer_init: dict = {},
        lr_scheduler_init: dict = {},
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_monitor: str = "train/total_loss_epoch",
    ):
        super().__init__(width, num_layers, ssl_layers, num_classes, optimizer_init, lr_scheduler_init, lr_scheduler_interval, lr_scheduler_monitor)
