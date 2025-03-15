
from typing import List

import cv2
import numpy as np
from torch import Tensor
import torch
from deep_sort_reid.types.detection import Detection
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image


def extract_features_resnet(file_input_path: str,
                            frames_detections: List[List[Detection]],
                            verbose=False
                            ) -> List[List[Tensor]]:

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    preprocess = weights.transforms()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    feature_extractor.to(device)

    cap = cv2.VideoCapture(file_input_path)

    if not cap.isOpened():
        print("Can't open video file")

    frame_idx = 0
    frames_features: List[List[Tensor]] = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if verbose:
            print("EXTRACTING FEATURES FRAME IDX ", frame_idx)
        detections = frames_detections[frame_idx]

        batch = []
        for detection in detections:
            # Images are indexed as (H, W, C)
            cropped_img = frame[int(detection.coords.start_y):int(detection.coords.end_y),
                                int(detection.coords.start_x):int(detection.coords.end_x)]

            cropped_img = Image.fromarray(cropped_img)
            model_input = (preprocess(
                cropped_img).unsqueeze(0)).to(device)
            batch.append(model_input)

        features = []
        if batch:
            batch = torch.cat(batch)
            object_features: Tensor = (feature_extractor(batch))

            for feature in object_features:
                features.append((feature.detach()).flatten())

        frames_features.append(features)
        frame_idx += 1

    return frames_features
