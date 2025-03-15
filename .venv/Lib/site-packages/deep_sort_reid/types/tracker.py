

from typing import List, NewType, Tuple
from pydantic import BaseModel, ConfigDict
from deep_sort_reid.enums.tracker import TrackState
from deep_sort_reid.types.coords import CoordinatesXYXY

TrackID = NewType('TrackID', int)

class TrackResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    track_id: TrackID
    cls: List[int]  # Classes separated by commas
    frame_idx: int
    coords: CoordinatesXYXY
    state: TrackState
    color: Tuple
