#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Dict, TypeVar

from flash.core.data.io.input import DataKeys
from flash.core.model import OutputKeys
from flash.core.utilities.stages import RunningStage


STAGE_TYPE = TypeVar("STAGE_TYPE", RunningStage, str)
InputDict = Dict[DataKeys, Any]
OutputDict = Dict[OutputKeys, Any]
