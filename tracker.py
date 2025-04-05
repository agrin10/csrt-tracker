import cv2
import logging
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SimpleTracker:
    def __init__(self, video_path):
        """Initialize the tracker with a video path."""
        self.video_path = video_path

        # Initialize YOLO model
        self.yolo_model = YOLO("yolov8n.pt")
        self.yolo_model.fuse()

        # Initialize DeepSORT tracker
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nn_budget=100,
            max_cosine_distance=0.4,
            max_iou_distance=0.7
        )

    def run(self):
        """Run the tracker on the video."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video file: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = max(1, int(1000 / fps)) if fps > 0 else 30

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.info("End of video or failed to read frame.")
                break

            # Perform object detection
            detections = self.detect_objects(frame)

            # Update tracker with detections
            tracked_objects = self.tracker.update_tracks(detections, frame=frame)

            # Draw tracking results
            self.draw_tracks(frame, tracked_objects)

            # Display the frame
            cv2.imshow("Simple Tracker", frame)
            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def detect_objects(self, frame):
        """Detect objects in the frame using YOLO."""
        results = self.yolo_model(frame, imgsz=640, conf=0.4, iou=0.4)[0]
        detections = []

        for det in results.boxes.data:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if int(cls) == 0:  # Only process people
                w, h = x2 - x1, y2 - y1
                bbox = [x1, y1, w, h]
                detections.append((bbox, conf, None))  # No feature embedding needed

        return detections

    def draw_tracks(self, frame, tracked_objects):
        """Draw tracked objects on the frame."""
        for track in tracked_objects:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Display track ID
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

if __name__ == "__main__":
    video_path = "E:\\Double1.mp4"  # Replace with your video path
    tracker = SimpleTracker(video_path)
    tracker.run()
