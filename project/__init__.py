#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .core.task import Task, MultiTask
from .image.classification.task import ImageClassifier
from .image.mim.task import MaskedImageModeling

__all__ = ["Task", "MultiTask", "ImageClassifier", "MaskedImageModeling"]
