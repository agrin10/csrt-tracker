

from enum import Enum


class TrackState(Enum):
    UNCONFIRMED = "UNCONFIRMED"
    CONFIRMED = "CONFIRMED"
    OUT_OF_FRAME = "OUT_OF_FRAME"
    DELETED = "DELETED"
    MASKED = "MASKED"
