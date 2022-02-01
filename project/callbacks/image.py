#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Optional, Tuple, Union, cast

import torch
import wandb
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY

from ..metrics import QueueStateCollection
from ..structs import BinaryPrediction, I, Mode, MultiClassPrediction, O, ResizeMixin
from .base import QueuedLoggingCallback


# TODO check if images from multiple GPUs end up in a single shared queue
@CALLBACK_REGISTRY
class ImageLoggingCallback(QueuedLoggingCallback[I, O]):
    queues: QueueStateCollection

    def __init__(
        self,
        name: str,
        queue_size: int,
        modes: Iterable[Union[str, Mode]] = ["val", "test"],
        max_size: Optional[Tuple[int, int]] = None,
        flush_interval: int = 0,
        negate_priority: bool = False,
    ):
        self.max_size = max_size
        super().__init__(name, queue_size, modes, flush_interval, negate_priority)

    @classmethod
    @torch.no_grad()
    def get_priority(cls, example: I, pred: O) -> Optional[Union[int, float]]:
        r"""Compute a priority for an example/prediction pair. When logging with a finite
        sized priority queue, only the ``len(queue)`` highest priority images will be logged.
        Typically priority would be assigned based on some metric (loss, entropy, error, etc.).
        """
        if example.is_batched:
            raise ValueError("`example` must be unbatched")
        elif pred.is_batched:
            raise ValueError("`pred` must be unbatched")

        if not example.has_label:
            return None
        assert example.label is not None

        if isinstance(pred, BinaryPrediction):
            return (example.label - pred.probs).abs_().item()
        elif isinstance(pred, MultiClassPrediction):
            scores = pred.probs_for_class(example.label)
            return (1.0 - scores).item()
        else:
            raise NotImplementedError(f"`get_priority` not implemented for type {type(pred)}")

    def caption(self, example: I, pred: O) -> Optional[str]:
        if example.has_label:
            assert example.label is not None
            if isinstance(pred, MultiClassPrediction):
                p_cls = pred.probs_for_class(example.label).item()
            else:
                p_cls = pred.probs.item()
            caption = f"P={p_cls:0.3f}"

        else:
            caption = f"P={pred.probs.argmax():0.3f}"

        return caption

    @torch.no_grad()
    def prepare_target(self, example: I, pred: O) -> wandb.Image:
        r"""Converts a raw example/prediction pair into an object to be logged"""
        # resize as needed
        if self.max_size is not None and isinstance(example, ResizeMixin):
            example = example.resize_to_fit(self.max_size, mode="bilinear", align_corners=False)
        if self.max_size is not None and isinstance(pred, ResizeMixin):
            pred = cast(O, cast(ResizeMixin, pred).resize_to_fit(self.max_size, mode="bilinear", align_corners=False))

        caption = self.caption(example, pred)
        return wandb.Image(example.img, caption=caption)
