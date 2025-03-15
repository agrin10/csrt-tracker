

from typing import List

from ultralytics import YOLO

from deep_sort_reid.types.coords import CoordinatesXYXY
from deep_sort_reid.types.detection import Detection
from deep_sort_reid.utils.misc import get_device


def detect_objects_yolo(file_input_path: str, model: YOLO, model_params={
                        "stream": True,
                        "classes": [0],
                        "iou": 0.5,
                        "save": False,
                        "conf": 0.5,
                        "device": get_device()
                        }) -> List[List[Detection]]:
    """
    A helper function for YOLO object detection

    Returns 2d list of detections, where first dimension is frames and second is detection
    """

    results = model(file_input_path, **model_params)

    frames_detections: List[List[Detection]] = []
    for frame_objects in results:

        detections: List[Detection] = []
        for object in frame_objects:

            startX, startY, endX, endY = object.boxes.xyxy[0].tolist()
            coords = CoordinatesXYXY(
                start_x=int(startX),
                start_y=int(startY),
                end_x=int(endX),
                end_y=int(endY))

            detection_model = Detection(
                coords=coords, cls=[int(object.boxes.cls.item())],
                confidence=object.boxes.conf, feature=None)

            detections.append(detection_model)
        frames_detections.append(detections)

    return frames_detections
