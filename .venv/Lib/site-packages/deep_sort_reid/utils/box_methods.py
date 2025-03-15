

"""
From to box formats
"""
from torch import Tensor
from deep_sort_reid.types.coords import CoordinatesXYAH
from deep_sort_reid.types.coords import CoordinatesXYXY


def from_xyah_to_tensor(box: CoordinatesXYAH) -> Tensor:
    return Tensor([box.center_x, box.center_y, box.aspect_ratio, box.height])


def from_xyxy_to_xyah(box: CoordinatesXYXY) -> CoordinatesXYAH:

    right_x = max(box.start_x, box.end_x)
    left_x = min(box.start_x, box.end_x)
    top_y = min(box.start_y, box.end_y)
    bottom_y = max(box.start_y, box.end_y)

    width = right_x - left_x
    height = bottom_y - top_y
    center_x = left_x + width / 2
    center_y = top_y + height / 2
    aspect_ratio = width / height

    return CoordinatesXYAH(center_x=center_x,
                           center_y=center_y,
                           aspect_ratio=aspect_ratio,
                           height=height)
