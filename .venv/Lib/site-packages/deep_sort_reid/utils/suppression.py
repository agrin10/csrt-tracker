

from typing import List

import numpy as np
from deep_sort_reid.metric.IouMetric import IouMetric
from deep_sort_reid.types.detection import Detection


def non_max_suppression(detections: List[List[Detection]],
                        max_overlap: float,
                        confidence_dependent: bool = True
                        ):
    """
    Perform non_max_surpression for boxes of same frame
    with optional confidence dependency
    Return detections, features where indices for every frame has been removed
    """
    detections_to_keep = []

    for frame_idx, detections_frame in enumerate(detections):
        detections_frame_to_keep = []
        detections_frame_idx = []
        if confidence_dependent:
            scores = []
            for detection in detections_frame:
                scores.append(detection.confidence)

            detections_frame_idx = np.flip(np.argsort(scores))

        else:
            for detection_idx, _ in enumerate(detections_frame):
                detections_frame_idx.append(detection_idx)

        for curr_idx, curr_det_idx in enumerate(detections_frame_idx):
            curr_det = detections_frame[curr_det_idx]
            for next_idx in range(len(detections_frame_idx) - 1, curr_idx, -1):
                next_det = detections_frame[detections_frame_idx[next_idx]]

                iou = IouMetric.iou(curr_det.coords, next_det.coords)
                if iou > max_overlap:
                    detections_frame_idx = np.delete(
                        detections_frame_idx, next_idx)

        for det_idx in detections_frame_idx:
            detections_frame_to_keep.append(detections_frame[det_idx])

        detections_to_keep.append(detections_frame_to_keep)

    return detections_to_keep
