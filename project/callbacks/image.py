#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import TYPE_CHECKING, ForwardRef, Iterable, List, Optional, Tuple, Type, TypeVar, Union, cast

import torch
import wandb
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor

from ..metrics import QueueStateCollection
from ..structs import BinaryPrediction, I, Mode, MultiClassPrediction, O, ResizeMixin
from .base import LoggingTarget, QueuedLoggingCallback


T = TypeVar("T", bound="ImageTarget")


if TYPE_CHECKING:
    from ..model.base import BaseModel
else:
    BaseModel = ForwardRef("BaseModel")


@dataclass
class ImageTarget(LoggingTarget[I, O]):
    example: I
    pred: O

    def overlay(self: T, overlay: Tensor, alpha: float = 0.5) -> T:
        raise NotImplementedError("overlay")
        assert overlay.ndim == self.example.img.ndim
        if self.example.is_batched:
            assert overlay.ndim == 4

    @property
    def caption(self) -> Optional[str]:
        if self.example.has_label:
            assert self.example.label is not None
            if isinstance(self.pred, MultiClassPrediction):
                p_cls = self.pred.probs_for_class(self.example.label).item()
            else:
                p_cls = self.pred.probs.item()
            caption = f"P={p_cls:0.3f}"

        else:
            caption = f"P={self.pred.probs.argmax():0.3f}"

        return caption

    @classmethod
    def create(cls: Type[T], example: I, pred: O) -> T:
        return cls(example, pred)


@dataclass
class WandBImageTarget(ImageTarget[I, O]):
    @rank_zero_only
    def log(
        self,
        pl_module: BaseModel,
        tag: str,
        step: int,
    ) -> wandb.Image:
        return wandb.Image(self.example.img, caption=self.caption)

    @classmethod
    def deferred_log(
        cls,
        pl_module: BaseModel,
        tag: str,
        step: int,
        targets: List[wandb.Image],
    ) -> None:
        target = {"trainer/global_step": step, tag: targets}
        if not pl_module.state.sanity_checking:
            pl_module.logger.experiment.log(target, commit=False)

    @classmethod
    def create(cls: Type[T], example: I, pred: O) -> T:
        return cls(example, pred)


class QueuedImageLoggingCallback(QueuedLoggingCallback[I, O]):
    queues: QueueStateCollection

    def __init__(
        self,
        name: str,
        queue_size: int,
        modes: Iterable[Mode] = [Mode.VAL, Mode.TEST],
        max_size: Optional[Tuple[int, int]] = None,
        target_cls: Type[ImageTarget] = WandBImageTarget,
        flush_interval: int = 0,
        negate_priority: bool = False,
    ):
        self.queue_size = queue_size
        self.max_size = max_size
        self.name = name
        self.queues = QueueStateCollection()
        super().__init__(name, modes, queue_size, target_cls, flush_interval, negate_priority)

    @classmethod
    @torch.no_grad()
    def get_priority(cls, example: I, pred: O) -> Union[int, float]:
        r"""Compute a priority for an example/prediction pair. When logging with a finite
        sized priority queue, only the ``len(queue)`` highest priority images will be logged.
        Typically priority would be assigned based on some metric (loss, entropy, error, etc.).
        """
        if example.is_batched:
            raise ValueError("`example` must be unbatched")
        elif pred.is_batched:
            raise ValueError("`pred` must be unbatched")

        if not example.has_label:
            return 0
        assert example.label is not None

        if isinstance(pred, BinaryPrediction):
            return (example.label - pred.probs).abs_().item()
        elif isinstance(pred, MultiClassPrediction):
            scores = pred.probs_for_class(example.label)
            return (1.0 - scores).item()
        else:
            raise NotImplementedError(f"`get_priority` not implemented for type {type(pred)}")

    @torch.no_grad()
    def prepare_logging_target(self, example: I, pred: O) -> ImageTarget[I, O]:
        r"""Converts a raw example/prediction pair into an object to be logged"""
        # resize as needed
        if self.max_size is not None and isinstance(example, ResizeMixin):
            example = example.resize_to_fit(self.max_size, mode="bilinear", align_corners=False)
        if self.max_size is not None and isinstance(pred, ResizeMixin):
            pred = cast(O, cast(ResizeMixin, pred).resize_to_fit(self.max_size, mode="bilinear", align_corners=False))

        return self.target_cls.create(example, pred)  # type: ignore
