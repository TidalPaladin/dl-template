#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod

from ..structs import Example, Prediction


class DataclassMetricMixin:
    r"""Mixin for :class`tm.Metric` subclasses that implement :func:`update` using
    :class:`Prediction` and :class:`Example` dataclasses as input
    """

    @abstractmethod
    def update(self, example: Example, pred: Prediction) -> None:
        ...
