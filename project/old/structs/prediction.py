#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractproperty
from dataclasses import dataclass
from typing import Iterable, Type, TypeVar

import torch
from torch import Tensor

from combustion.lightning.metrics import Entropy
from combustion.util.dataclasses import BatchMixin, TensorDataclass


O = TypeVar("O", bound="Prediction")


@dataclass(repr=False, eq=False)
class Prediction(TensorDataclass, BatchMixin):
    r"""Base container for model outputs."""
    logits: Tensor

    __slice_fields__ = ["logits"]

    def __eq__(self, other: "Prediction") -> bool:
        return torch.allclose(self.logits, other.logits)

    @abstractproperty
    def probs(self) -> Tensor:
        return self.logits.float().softmax(dim=-1)

    @property
    def is_batched(self) -> bool:
        return self.logits.ndim >= 2

    @classmethod
    def from_unbatched(cls: Type[O], predictions: Iterable[O]) -> O:
        logits = torch.stack([pred.logits for pred in predictions], dim=0)
        return cls(logits)

    def __len__(self) -> int:
        if not self.is_batched:
            return 0
        return self.logits.shape[0]


@dataclass(repr=False, eq=False)
class BinaryPrediction(Prediction):
    def __post_init__(self):
        if not 1 <= self.logits.ndim <= 2:
            raise ValueError(f"Unexpected shape {self.logits.shape} for `logits`")

    @property
    def probs(self) -> Tensor:
        return self.logits.float().sigmoid()

    def classes(self, threshold: float) -> Tensor:
        return (self.probs >= threshold).long()

    @property
    def entropy(self) -> Tensor:
        return Entropy.compute_binary_entropy(self.logits.float(), from_logits=True)


@dataclass(repr=False, eq=False)
class MultiClassPrediction(Prediction):
    def __post_init__(self):
        if not 1 <= self.logits.ndim <= 2:
            raise ValueError(f"Unexpected shape {self.logits.shape} for `logits`")
        if self.logits.shape[-1] <= 1:
            raise ValueError(f"Invalid logits shape for multiclass predicition: {self.logits.shape}")

    @property
    def probs(self) -> Tensor:
        return self.logits.float().softmax(dim=-1)

    @property
    def classes(self) -> Tensor:
        return self.logits.argmax(dim=-1)

    @property
    def num_classes(self) -> int:
        return self.logits.shape[-1]

    @property
    def entropy(self) -> Tensor:
        return Entropy.compute_categorical_entropy(self.logits.float(), from_logits=True)

    def probs_for_class(self, classes: Tensor) -> Tensor:
        if (ndim_diff := self.logits.ndim - classes.ndim) == 1:
            classes = classes.unsqueeze(-1)
        elif abs(ndim_diff) >= 1:
            raise ValueError(f"Shape mismatch: {self.logits.shape} vs {classes.shape}")

        assert self.logits.ndim == classes.ndim
        assert classes.shape[-1] == 1
        if (classes >= self.num_classes).any():
            raise ValueError("`classes` contained an index greater than `self.num_classes`")
        return self.probs[..., classes]
