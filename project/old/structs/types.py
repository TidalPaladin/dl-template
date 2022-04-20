#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, List, Optional, Set, Union, TypeVar
from flash.core.utilities.stages import RunningStage


STAGE_TYPE = TypeVar("STAGE_TYPE", RunningStage, str)
