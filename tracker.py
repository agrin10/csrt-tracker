import threading
import cv2
import time
import logging
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import torch
import torchreid

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MultiCameraTracker:
    def __init__(self, video_paths):
        """Initialize multiple CameraTracker instances and manage tracking across them."""
        self.video_paths = video_paths
        self.cameras = {}
        self.global_tracker = DeepSort(max_age=10000, n_init=2)  # Global DeepSORT for re-identification
        self.yolo_model = YOLO("yolov8n.pt")  # Load YOLO model
        self.lock = threading.Lock()

        for camera_id, video_path in video_paths.items():
            self.cameras[camera_id] = CameraTracker(camera_id, video_path, self)

    def start_all(self):
        """Start tracking for all cameras."""
        for cam in self.cameras.values():
            cam.start()

    def stop_all(self):
        """Stop tracking for all cameras."""
        for cam in self.cameras.values():
            cam.stop()

class CameraTracker:
    def __init__(self, camera_id, video_path, parent_tracker):
        """Initialize a CameraTracker instance with YOLO and DeepSORT."""
        self.camera_id = camera_id
        self.video_path = video_path
        self.cap = None
        self.running = True
        self.thread = threading.Thread(target=self.track_camera, daemon=True)
        
        self.parent_tracker = parent_tracker
        self.yolo_model = parent_tracker.yolo_model  # Shared YOLO model
        self.global_tracker = parent_tracker.global_tracker  # Shared DeepSORT tracker
        
        self.selected_region = None
        self.tracking_started = False
        self.track_history = {}
    
    def track_camera(self):
        """Main tracking loop."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            logging.error(f"Failed to open video file: {self.video_path}")
            self.running = False
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_delay = int(1000 / fps) if fps > 0 else 30
        
        cv2.namedWindow(f"Camera {self.camera_id}")
        cv2.setMouseCallback(f"Camera {self.camera_id}", self.select_region)

        while self.running:
            success, frame = self.cap.read()
            if not success:
                logging.info(f"Restarting video for Camera {self.camera_id}")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video
                continue

            detections = self.detect_objects(frame) if self.tracking_started else []

            if detections:
                tracked_objects = self.global_tracker.update_tracks(detections, frame=frame)
                new_track_history = {}
                
                for track in tracked_objects:
                    if not track.is_confirmed():
                        continue
                    
                    track_id = track.track_id
                    bbox = track.to_tlbr()
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Store track history
                    new_track_history[track_id] = (x1, y1, x2, y2)
                    
                    # Draw bounding box and ID
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                self.track_history = new_track_history  # Update tracking history

            if self.selected_region is not None:
                x1, y1, x2, y2 = self.selected_region
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            cv2.imshow(f"Camera {self.camera_id}", frame)
            time.sleep(1 / fps)
            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyWindow(f"Camera {self.camera_id}")

    def detect_objects(self, frame):
        """Run YOLO detection and extract DeepSORT feature embeddings."""
        if self.selected_region is None:
            return []
        
        x1_s, y1_s, x2_s, y2_s = self.selected_region
        results = self.yolo_model(frame)[0]
        detections = []

        for det in results.boxes.data:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if int(cls) == 0:  # Class 0 is 'person'
                if x1 < x2_s and x2 > x1_s and y1 < y2_s and y2 > y1_s:
                    w, h = x2 - x1, y2 - y1
                    bbox = [x1, y1, w, h]
                    detections.append((bbox, conf, 'person'))
        
        return detections

    def select_region(self, event, x, y, flags, param):
        """Mouse callback function to select a region for object tracking."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.region_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            x1, y1 = self.region_start
            x2, y2 = x, y
            self.selected_region = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            self.tracking_started = True

    def start(self):
        """Start the tracking thread."""
        if not self.thread.is_alive():
            self.thread.start()

    def stop(self):
        """Stop the tracking thread."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        if self.cap and self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    video_paths = {
        # "cam1": "E:\\Double1.mp4",
        "cam2": "E:\\Single1.mp4",
    }

    tracker = MultiCameraTracker(video_paths)
    tracker.start_all()

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        tracker.stop_all()

    cv2.destroyAllWindows()