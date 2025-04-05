import threading
import cv2
import time
import logging
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import torch
import torchreid
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MultiCameraTracker:
    def __init__(self, video_paths):
        """Initialize multiple CameraTracker instances with cross-camera tracking."""
        self.video_paths = video_paths
        self.cameras = {}
        
        # Configure DeepSORT tracker
        self.global_tracker = DeepSort(
            max_age=30,
            n_init=3,
            nn_budget=100,
            max_cosine_distance=0.4,
            max_iou_distance=0.7
        )
        
        # Load YOLO model
        self.yolo_model = YOLO("yolov8n.pt")
        self.yolo_model.fuse()
        
        # Feature extractor for re-identification
        self.reid_model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )
        self.reid_model.eval()
        
        # Cross-camera tracking variables
        self.global_id_counter = 0
        self.global_tracks = {}  # {global_id: {'features': [], 'last_seen': timestamp, 'cameras': []}}
        self.lock = threading.Lock()
        
        # Initialize cameras
        for camera_id, video_path in video_paths.items():
            self.cameras[camera_id] = CameraTracker(
                camera_id, 
                video_path, 
                self,
                yolo_model=self.yolo_model,
                global_tracker=self.global_tracker,
                reid_model=self.reid_model
            )

    def get_next_global_id(self):
        """Generate unique global IDs for cross-camera tracking."""
        with self.lock:
            self.global_id_counter += 1
            return self.global_id_counter

    def match_across_cameras(self):
        """Match tracks across all cameras using appearance features."""
        current_time = time.time()
        all_active_tracks = []

        # Collect all active tracks from all cameras
        for cam_id, camera in self.cameras.items():
            for track_id, track_info in camera.track_history.items():
                if track_id in camera.feature_cache:
                    all_active_tracks.append({
                        'camera_id': cam_id,
                        'track_id': track_id,
                        'features': camera.feature_cache[track_id],
                        'bbox': track_info['bbox'],
                        'timestamp': current_time
                    })

        # Match tracks across cameras
        matched_pairs = []
        for i in range(len(all_active_tracks)):
            for j in range(i + 1, len(all_active_tracks)):
                track1 = all_active_tracks[i]
                track2 = all_active_tracks[j]

                # Skip if same camera
                if track1['camera_id'] == track2['camera_id']:
                    continue

                # Calculate similarity
                similarity = self.cosine_similarity(track1['features'], track2['features'])
                logging.info(f"Similarity between {track1['camera_id']}:{track1['track_id']} "
                            f"and {track2['camera_id']}:{track2['track_id']} = {similarity}")

                # Threshold for matching (adjust as needed)
                if similarity > 0.8:
                    matched_pairs.append((track1, track2))

        # Process matches and assign global IDs
        for track1, track2 in matched_pairs:
            with self.lock:
                # Case 1: Neither track has a global ID
                if 'global_id' not in self.cameras[track1['camera_id']].track_history[track1['track_id']] and \
                'global_id' not in self.cameras[track2['camera_id']].track_history[track2['track_id']]:
                    global_id = self.get_next_global_id()
                    self.cameras[track1['camera_id']].track_history[track1['track_id']]['global_id'] = global_id
                    self.cameras[track2['camera_id']].track_history[track2['track_id']]['global_id'] = global_id
                    self.global_tracks[global_id] = {
                        'features': track1['features'],
                        'last_seen': current_time,
                        'cameras': [track1['camera_id'], track2['camera_id']]
                    }
                    logging.info(f"Assigned new global ID {global_id} to tracks "
                                f"{track1['camera_id']}:{track1['track_id']} and {track2['camera_id']}:{track2['track_id']}")

                # Case 2: Track1 has global ID but track2 doesn't
                elif 'global_id' in self.cameras[track1['camera_id']].track_history[track1['track_id']] and \
                    'global_id' not in self.cameras[track2['camera_id']].track_history[track2['track_id']]:
                    global_id = self.cameras[track1['camera_id']].track_history[track1['track_id']]['global_id']
                    self.cameras[track2['camera_id']].track_history[track2['track_id']]['global_id'] = global_id
                    self.global_tracks[global_id]['last_seen'] = current_time
                    if track2['camera_id'] not in self.global_tracks[global_id]['cameras']:
                        self.global_tracks[global_id]['cameras'].append(track2['camera_id'])
                    logging.info(f"Assigned existing global ID {global_id} to track "
                                f"{track2['camera_id']}:{track2['track_id']}")

                # Case 3: Track2 has global ID but track1 doesn't
                elif 'global_id' not in self.cameras[track1['camera_id']].track_history[track1['track_id']] and \
                    'global_id' in self.cameras[track2['camera_id']].track_history[track2['track_id']]:
                    global_id = self.cameras[track2['camera_id']].track_history[track2['track_id']]['global_id']
                    self.cameras[track1['camera_id']].track_history[track1['track_id']]['global_id'] = global_id
                    self.global_tracks[global_id]['last_seen'] = current_time
                    if track1['camera_id'] not in self.global_tracks[global_id]['cameras']:
                        self.global_tracks[global_id]['cameras'].append(track1['camera_id'])
                    logging.info(f"Assigned existing global ID {global_id} to track "
                                f"{track1['camera_id']}:{track1['track_id']}")

    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two feature vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def start_all(self):
        """Start tracking for all cameras."""
        for cam in self.cameras.values():
            cam.start()

    def stop_all(self):
        """Stop tracking for all cameras."""
        for cam in self.cameras.values():
            cam.stop()

class CameraTracker:
    def __init__(self, camera_id, video_path, parent_tracker, yolo_model, global_tracker, reid_model):
        """Camera tracker with cross-camera capabilities."""
        self.camera_id = camera_id
        self.video_path = video_path
        self.cap = None
        self.running = True
        self.thread = threading.Thread(target=self.track_camera, daemon=True)
        
        # Shared components
        self.parent_tracker = parent_tracker
        self.yolo_model = yolo_model
        self.global_tracker = global_tracker
        self.reid_model = reid_model
        
        # Tracking state
        self.selected_regions = []  # Now supports multiple regions
        self.tracking_started = False
        self.track_history = {}  # Format: {track_id: {'bbox': (x1,y1,x2,y2), 'global_id': id}}
        self.lost_tracks = {}
        self.feature_cache = {}
        self.colors = np.random.rand(100, 3) * 255  # Enough colors for many tracks

    def check_lost_tracks(self):
        """Handle lost tracks and their re-identification."""
        current_time = time.time()
        tracks_to_remove = []

        # Check each lost track
        for track_id, track_info in self.lost_tracks.items():
            # Remove tracks that have been lost for too long (e.g., 30 seconds)
            if current_time - track_info['last_seen'] > 30:
                tracks_to_remove.append(track_id)
                continue

            # Try to match with current tracks in other cameras
            if track_info['global_id'] is not None:
                for cam_id, camera in self.parent_tracker.cameras.items():
                    if cam_id == self.camera_id:
                        continue

                    for current_track_id, current_track_info in camera.track_history.items():
                        if current_track_id in camera.feature_cache:
                            # Calculate similarity between lost track and current track
                            similarity = self.parent_tracker.cosine_similarity(
                                track_info['features'],
                                camera.feature_cache[current_track_id]
                            )

                            # If similarity is high enough, update global ID
                            if similarity > 0.8:
                                if 'global_id' not in current_track_info:
                                    current_track_info['global_id'] = track_info['global_id']
                                tracks_to_remove.append(track_id)
                                break

        # Remove expired tracks
        for track_id in tracks_to_remove:
            self.lost_tracks.pop(track_id, None)

    def track_camera(self):
        """Main tracking loop with cross-camera support."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            logging.error(f"Failed to open video file: {self.video_path}")
            self.running = False
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_delay = max(1, int(1000 / fps)) if fps > 0 else 30
        
        cv2.namedWindow(f"Camera {self.camera_id}")
        cv2.setMouseCallback(f"Camera {self.camera_id}", self.select_region)

        frame_count = 0
        prev_time = time.time()
        
        while self.running:
            success, frame = self.cap.read()
            if not success:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # Detection and tracking
            detections = self.detect_objects(frame) if self.tracking_started else []
            
            if detections:
                tracked_objects = self.global_tracker.update_tracks(detections, frame=frame)
                self.update_track_history(frame, tracked_objects)
                self.check_lost_tracks()
            
            # Periodically run cross-camera matching
            if frame_count % 30 == 0:  # Every 30 frames
                self.parent_tracker.match_across_cameras()
            
            # Visualization
            self.draw_tracking_info(frame, fps)
            
            # Draw all selected regions
            for region in self.selected_regions:
                x1, y1, x2, y2 = region
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            cv2.imshow(f"Camera {self.camera_id}", frame)
            
            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_tracking()
            elif key == ord('c'):  # Clear all selections
                self.selected_regions = []
                self.tracking_started = False
            
            frame_count += 1

        self.cap.release()
        cv2.destroyWindow(f"Camera {self.camera_id}")

    def detect_objects(self, frame):
        """Detect objects in all selected regions."""
        if not self.selected_regions:
            return []
        
        # Resize frame for faster processing
        scale = 0.5
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        results = self.yolo_model(small_frame, imgsz=640, conf=0.3, iou=0.4)[0]  # Lower confidence threshold
        detections = []

        for det in results.boxes.data:
            x1, y1, x2, y2, conf, cls = det.tolist()
            # Scale back to original size
            x1, y1, x2, y2 = map(lambda x: int(x/scale), [x1, y1, x2, y2])
            
            if int(cls) == 0:  # Only process people
                # Check if in any selected region
                in_any_region = False
                for region in self.selected_regions:
                    x1_s, y1_s, x2_s, y2_s = region
                    # Increase overlap area for better detection
                    if (x1 < x2_s + 10 and x2 > x1_s - 10 and y1 < y2_s + 10 and y2 > y1_s - 10):
                        in_any_region = True
                        break
                
                if in_any_region:
                    w, h = x2 - x1, y2 - y1
                    bbox = [x1, y1, w, h]
                    crop = frame[max(0, int(y1)):min(frame.shape[0], int(y2)), 
                                max(0, int(x1)):min(frame.shape[1], int(x2))]
                    features = self.extract_features(crop) if crop.size > 0 else None
                    if features is not None:  # Only add detection if features were extracted successfully
                        detections.append((bbox, conf, features))
        
        return detections

    def extract_features(self, image):
        """Extract re-identification features."""
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (128, 256))
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            
            with torch.no_grad():
                features = self.reid_model(image)
            return features.cpu().numpy().flatten()
        except Exception as e:
            logging.warning(f"Feature extraction failed: {str(e)}")
            return None

    def update_track_history(self, frame, tracked_objects):
        """Update track history with cross-camera support."""
        current_ids = set()
        new_track_history = {}
        
        for track in tracked_objects:
            if not track.is_confirmed() or track.time_since_update > 1:  # Skip unconfirmed or stale tracks
                continue

            track_id = str(track.track_id)
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            current_ids.add(track_id)
            
            # Initialize track info if not exists
            if track_id not in new_track_history:
                new_track_history[track_id] = {
                    'bbox': (x1, y1, x2, y2),
                    'global_id': self.track_history.get(track_id, {}).get('global_id', None),
                    'last_update': time.time()
                }
            
            # Update features only every few frames to improve performance
            if track_id not in self.feature_cache or \
               time.time() - new_track_history[track_id].get('last_update', 0) > 0.5:  # Update every 0.5 seconds
                crop = frame[max(0, y1):min(frame.shape[0], y2), 
                            max(0, x1):min(frame.shape[1], x2)]
                if crop.size > 0:
                    features = self.extract_features(crop)
                    if features is not None:
                        self.feature_cache[track_id] = features
                        new_track_history[track_id]['last_update'] = time.time()
        
        # Handle lost tracks - only consider tracks lost if missing for more than 2 frames
        lost_ids = set()
        for track_id in self.track_history:
            if track_id not in current_ids and \
               time.time() - self.track_history[track_id].get('last_update', 0) > 0.1:  # 100ms threshold
                lost_ids.add(track_id)
        
        for track_id in lost_ids:
            if track_id in self.feature_cache:
                self.lost_tracks[track_id] = {
                    'last_seen': time.time(),
                    'last_position': self.track_history[track_id]['bbox'],
                    'features': self.feature_cache[track_id],
                    'global_id': self.track_history[track_id].get('global_id', None)
                }
        
        self.track_history = new_track_history

    def draw_tracking_info(self, frame, fps):
        """Draw tracking information with global IDs."""
        # Draw active tracks
        for track_id, track_info in list(self.track_history.items()):
            try:
                x1, y1, x2, y2 = track_info['bbox']
                color = self.colors[int(track_id) % len(self.colors)]
                
                # Draw bounding box with thicker lines
                cv2.rectangle(frame, (x1, y1), (x2, y2), color.tolist(), 3)
                
                # Display ID information with better visibility
                display_text = f"ID: {track_id}"
                if 'global_id' in track_info and track_info['global_id'] is not None:
                    display_text = f"Global: {track_info['global_id']}"  # Prioritize global ID
                
                # Add background to text for better visibility
                text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_size[0], y1), color.tolist(), -1)
                cv2.putText(frame, display_text, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            except Exception as e:
                logging.warning(f"Error drawing track {track_id}: {e}")

        # Draw lost tracks with different style (only if recently lost)
        current_time = time.time()
        for track_id, track_info in list(self.lost_tracks.items()):
            if current_time - track_info['last_seen'] <= 1.0:  # Only show recently lost tracks
                try:
                    x1, y1, x2, y2 = track_info['last_position']
                    color = (0, 0, 255)  # Red for lost tracks
                    
                    # Draw dashed rectangle for lost tracks
                    for i in range(0, 8):  # Increase number of dashes
                        pt1 = (x1 + i * (x2 - x1) // 8, y1)
                        pt2 = (x1 + (i + 1) * (x2 - x1) // 8, y1)
                        cv2.line(frame, pt1, pt2, color, 2)
                        pt1 = (x1 + i * (x2 - x1) // 8, y2)
                        pt2 = (x1 + (i + 1) * (x2 - x1) // 8, y2)
                        cv2.line(frame, pt1, pt2, color, 2)
                        pt1 = (x1, y1 + i * (y2 - y1) // 8)
                        pt2 = (x1, y1 + (i + 1) * (y2 - y1) // 8)
                        cv2.line(frame, pt1, pt2, color, 2)
                        pt1 = (x2, y1 + i * (y2 - y1) // 8)
                        pt2 = (x2, y1 + (i + 1) * (y2 - y1) // 8)
                        cv2.line(frame, pt1, pt2, color, 2)
                    
                    # Display lost track information
                    display_text = f"Lost: {track_id}"
                    if track_info['global_id'] is not None:
                        display_text = f"Lost Global: {track_info['global_id']}"  # Prioritize global ID
                    
                    text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_size[0], y1), color, -1)
                    cv2.putText(frame, display_text, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                except Exception as e:
                    logging.warning(f"Error drawing lost track {track_id}: {e}")

        # Display FPS and status with better visibility
        fps_color = (0, 255, 0) if fps > 10 else (0, 165, 255) if fps > 5 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        cv2.putText(frame, f"Active Tracks: {len(self.track_history)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Lost Tracks: {len(self.lost_tracks)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def select_region(self, event, x, y, flags, param):
        """Mouse callback for multiple region selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.region_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            x1, y1 = self.region_start
            x2, y2 = x, y
            new_region = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            
            # Check if we should remove an existing region
            for i, region in enumerate(self.selected_regions):
                if self.regions_overlap(new_region, region):
                    self.selected_regions.pop(i)
                    return
            
            # Add new region
            self.selected_regions.append(new_region)
            self.tracking_started = True

    def regions_overlap(self, region1, region2):
        """Check if two regions overlap significantly."""
        x1_1, y1_1, x2_1, y2_1 = region1
        x1_2, y1_2, x2_2, y2_2 = region2
        
        # Calculate intersection area
        dx = min(x2_1, x2_2) - max(x1_1, x1_2)
        dy = min(y2_1, y2_2) - max(y1_1, y1_2)
        
        return dx > 0 and dy > 0 and (dx * dy) > 0.5 * min(
            (x2_1 - x1_1) * (y2_1 - y1_1),
            (x2_2 - x1_2) * (y2_2 - y1_2)
        )

    def reset_tracking(self):
        """Reset tracking state."""
        self.track_history = {}
        self.lost_tracks = {}
        self.feature_cache = {}
        logging.info(f"Tracking reset for camera {self.camera_id}")

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
        "cam1": "E:\\Double1.mp4",
        "cam2": "E:\\Single1.mp4",
    }

    tracker = MultiCameraTracker(video_paths)
    tracker.start_all()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        tracker.stop_all()

    cv2.destroyAllWindows() 