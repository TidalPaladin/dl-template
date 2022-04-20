#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, List, Optional, Set, Union, TypeVar
from flash.core.utilities.stages import RunningStage
from flash.core.model import OutputKeys
from flash.core.data.io.input import DataKeys
from typing import Dict, Any


STAGE_TYPE = TypeVar("STAGE_TYPE", RunningStage, str)
InputDict = Dict[DataKeys, Any]
OutputDict = Dict[OutputKeys, Any]

