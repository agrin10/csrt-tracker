

from typing import List
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from deep_sort_reid.types.coords import CoordinatesXYXY


class Detection(BaseModel):
    model_config = ConfigDict(from_attributes=True,
                              arbitrary_types_allowed=True)

    cls: List[int]  # Classes separated by commas
    coords: CoordinatesXYXY
    confidence: float
    feature: Tensor | None
