import cv2
import numpy as np
import logging
import os
import sys
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("tracker_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ObjectTracker")

class SimpleReID(nn.Module):
    def __init__(self):
        super(SimpleReID, self).__init__()
        # Use ResNet18 as backbone for feature extraction
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # Remove the last fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        # Add a new FC layer for ReID features
        self.fc = nn.Linear(512, 256)
        
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return nn.functional.normalize(x, p=2, dim=1)  # L2 normalize the features

class DrawTracker:
    def __init__(self, video_sources, yolo_model="yolov8n.pt", yolo_conf=0.5):
        """Initialize the tracker with multiple video sources."""
        self.video_sources = video_sources
        self.yolo_model = yolo_model
        self.yolo_conf = yolo_conf
        
        # Initialize YOLO model
        logger.info(f"Loading YOLO model: {yolo_model}")
        self.yolo = YOLO(yolo_model)
        logger.info(f"YOLO model loaded: {yolo_model}")
        
        # Initialize ReID model
        logger.info("Initializing ReID model...")
        self.reid_model = SimpleReID()
        if torch.cuda.is_available():
            self.reid_model = self.reid_model.cuda()
        self.reid_model.eval()
        logger.info("ReID model initialized")
        
        # ReID transform
        self.reid_transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Store features of tracked object
        self.tracked_features = None
        self.reid_threshold = 0.90  # Increased threshold for high confidence matches
        self.recovery_threshold = 0.85  # Increased recovery threshold
        self.show_threshold = 0.70  # Threshold for showing matches at all
        
        # Add debug mode
        self.debug_mode = True
        
        # Initialize video captures
        self.caps = []
        for source in video_sources:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logger.error(f"Failed to open video source: {source}")
                continue
            self.caps.append(cap)
            logger.info(f"Opened video source: {source}")
        
        if not self.caps:
            raise ValueError("No valid video sources found")
        
        # Get video properties
        self.fps = min(cap.get(cv2.CAP_PROP_FPS) for cap in self.caps)
        if self.fps <= 0:
            self.fps = 30
        logger.info(f"Using FPS: {self.fps}")
        
        # Tracking state
        self.running = False
        self.paused = False
        self.tracking_enabled = False
        self.tracker = None  # Single tracker for the selected camera
        self.tracking_camera = None  # Which camera is being tracked
        
        # Drawing state
        self.drawing = False
        self.roi = None
        self.start_point = None
        self.end_point = None
        self.current_camera = None
        
        # Add tracking state variables
        self.lost_track_frames = 0
        self.max_lost_frames = 30  # Increased from 10 to 30 frames
        self.last_valid_bbox = None  # Store last valid bounding box
        self.track_lost = False  # Flag to indicate if track is lost
        
        # Add confidence tracking
        self.consecutive_matches = 0
        self.min_consecutive_matches = 3  # Require multiple consecutive matches before switching
        self.last_match_camera = None
        self.last_match_box = None
        
        # Add tracking recovery parameters
        self.recovery_roi_scale = 1.5  # Scale factor for search region when track is lost
        self.last_known_size = None  # Store last known object size for recovery
        self.last_known_position = None  # Store last known position for recovery
        
        # Track highest confidence seen
        self.highest_confidence_seen = 0.0
        self.highest_confidence_camera = None
        self.highest_confidence_box = None
    
    def extract_reid_features(self, frame, bbox):
        """Extract ReID features from a person bounding box"""
        try:
            x, y, w, h = map(int, bbox)
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                logger.warning("Invalid bounding box coordinates for ReID")
                return None
            
            # Extract person ROI
            person_roi = frame[y:y+h, x:x+w]
            if person_roi.size == 0:
                logger.warning("Empty ROI for ReID")
                return None
            
            # Convert to PIL Image
            person_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            person_pil = Image.fromarray(person_roi)
            
            # Apply transforms
            person_tensor = self.reid_transform(person_pil).unsqueeze(0)
            if torch.cuda.is_available():
                person_tensor = person_tensor.cuda()
            
            # Extract features
            with torch.no_grad():
                features = self.reid_model(person_tensor)
            
            return features.cpu()
        except Exception as e:
            logger.error(f"Error extracting ReID features: {str(e)}")
            return None
    
    def compute_similarity(self, features1, features2):
        """Compute cosine similarity between two feature vectors"""
        if features1 is None or features2 is None:
            return 0.0
        
        try:
            # Normalize features before computing similarity
            features1 = torch.nn.functional.normalize(features1, p=2, dim=1)
            features2 = torch.nn.functional.normalize(features2, p=2, dim=1)
            
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(features1, features2)
            return similarity.item()
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return 0.0
    
    def detect_person_in_frame(self, frame):
        """Detect persons in the frame using YOLO"""
        try:
            results = self.yolo(frame, conf=self.yolo_conf, classes=[0])  # class 0 is person
            boxes = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = x2 - x1
                    h = y2 - y1
                    boxes.append((int(x1), int(y1), int(w), int(h)))
            return boxes
        except Exception as e:
            logger.error(f"Error detecting persons: {str(e)}")
            return []
    
    def update_tracking_state(self, frame, bbox):
        """Update tracking state with new detection"""
        x, y, w, h = [int(v) for v in bbox]
        self.last_known_position = (x + w//2, y + h//2)  # Center point
        self.last_known_size = (w, h)
        self.last_valid_bbox = bbox
        
        # Extract new ReID features
        new_features = self.extract_reid_features(frame, bbox)
        if new_features is not None:
            # Update features with moving average
            if self.tracked_features is None:
                self.tracked_features = new_features
            else:
                alpha = 0.7  # Weight for current features
                self.tracked_features = alpha * self.tracked_features + (1 - alpha) * new_features

    def recover_lost_track(self, frame, camera_idx):
        """Attempt to recover lost track using last known position and size"""
        if self.last_known_position is None or self.last_known_size is None:
            return None, 0.0

        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        center_x, center_y = self.last_known_position
        w, h = self.last_known_size

        # Calculate search region (expanded)
        search_w = int(w * self.recovery_roi_scale)
        search_h = int(h * self.recovery_roi_scale)
        search_x = max(0, center_x - search_w//2)
        search_y = max(0, center_y - search_h//2)
        search_w = min(search_w, frame_width - search_x)
        search_h = min(search_h, frame_height - search_y)

        # Detect persons in frame
        detected_boxes = self.detect_person_in_frame(frame)
        best_match = None
        best_similarity = 0

        for box in detected_boxes:
            # Check if detection is in search region
            x, y, w, h = box
            box_center_x = x + w//2
            box_center_y = y + h//2
            
            if (abs(box_center_x - center_x) < search_w//2 and 
                abs(box_center_y - center_y) < search_h//2):
                
                features = self.extract_reid_features(frame, box)
                if features is not None:
                    similarity = self.compute_similarity(self.tracked_features, features)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = box

        return best_match, best_similarity

    def run(self):
        """Run the tracker on multiple video streams."""
        self.running = True
        
        # Print instructions
        print("\n=== Multi-Camera Person Tracking System ===")
        print("Controls:")
        print("  'p' - Pause/Resume video playback")
        print("  Click - Select object to track (video must be paused)")
        print("  'q' - Quit the application")
        print("\nTo select an object to track:")
        print("1. Press 'p' to pause the video")
        print("2. Click near the object you want to track")
        print("3. Press 'p' again to resume tracking")
        print("\nVideos are playing. Press 'p' to pause and select an object.")
        print("=====================================\n")
        
        # Create windows for each camera
        for i, source in enumerate(self.video_sources):
            window_name = f"Camera {i+1} - {os.path.basename(source)}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            cv2.setMouseCallback(window_name, self.mouse_callback, param=i)
        
        # Main loop
        frames = [None] * len(self.caps)
        while self.running:
            # Process each frame
            for i, cap in enumerate(self.caps):
                if self.paused:
                    # If paused, show the last frame with a "PAUSED" text
                    if frames[i] is not None:
                        frame_with_text = frames[i].copy()
                        cv2.putText(frame_with_text, "PAUSED", (50, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        cv2.imshow(f"Camera {i+1} - {os.path.basename(self.video_sources[i])}", frame_with_text)
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    # End of video, restart
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        logger.error(f"Failed to read frame from camera {i}")
                        continue
                
                frames[i] = frame.copy()
                
                # Process frame for tracking
                if self.tracking_enabled:
                    if i == self.tracking_camera:
                        if self.track_lost:
                            # Try to recover track
                            recovered_box, similarity = self.recover_lost_track(frame, i)
                            
                            if recovered_box is not None and similarity > self.recovery_threshold:
                                # Reinitialize tracker with recovered box
                                self.tracker = cv2.legacy.TrackerCSRT_create()
                                success = self.tracker.init(frame, recovered_box)
                                
                                if success:
                                    self.track_lost = False
                                    self.lost_track_frames = 0
                                    self.update_tracking_state(frame, recovered_box)
                                    
                                    x, y, w, h = recovered_box
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                    cv2.putText(frame, f"Track Recovered ({similarity*100:.1f}%)", 
                                              (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            # Normal tracking update
                            success, bbox = self.tracker.update(frame)
                            
                            if success:
                                # Update tracking state
                                self.update_tracking_state(frame, bbox)
                                self.lost_track_frames = 0
                                
                                # Draw tracking box
                                (x, y, w, h) = [int(v) for v in bbox]
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(frame, "Tracking", (x, y - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            else:
                                # Try immediate recovery before counting as lost frame
                                recovered_box, similarity = self.recover_lost_track(frame, i)
                                
                                if recovered_box is not None and similarity > self.recovery_threshold:
                                    # Reinitialize tracker with recovered box
                                    self.tracker = cv2.legacy.TrackerCSRT_create()
                                    success = self.tracker.init(frame, recovered_box)
                                    
                                    if success:
                                        self.update_tracking_state(frame, recovered_box)
                                        x, y, w, h = recovered_box
                                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                        cv2.putText(frame, f"Track Stabilized ({similarity*100:.1f}%)", 
                                                  (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    else:
                                        self.lost_track_frames += 1
                                else:
                                    self.lost_track_frames += 1
                                
                                if self.lost_track_frames >= self.max_lost_frames:
                                    self.track_lost = True
                                    logger.info(f"Lost track in camera {i+1}, switching to search mode")
                                    cv2.putText(frame, "Track Lost - Searching", (30, 30),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        # Search for matches in other cameras
                        detected_boxes = self.detect_person_in_frame(frame)
                        best_match = None
                        best_similarity = 0
                        
                        # Store all matches for display
                        all_matches = []
                        
                        for box in detected_boxes:
                            features = self.extract_reid_features(frame, box)
                            if features is not None:
                                similarity = self.compute_similarity(self.tracked_features, features)
                                all_matches.append((box, similarity))
                                
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match = (box, similarity)
                                    
                                    # Update highest confidence seen
                                    if similarity > self.highest_confidence_seen:
                                        self.highest_confidence_seen = similarity
                                        self.highest_confidence_camera = i
                                        self.highest_confidence_box = box
                                
                                # Draw all detection boxes with scores in debug mode
                                if self.debug_mode and similarity > self.show_threshold:
                                    x, y, w, h = box
                                    # Color based on similarity threshold
                                    if similarity > self.reid_threshold:
                                        color = (0, 255, 0)  # Green for high confidence
                                        thickness = 3  # Thicker box for high confidence
                                    elif similarity > 0.8:
                                        color = (0, 165, 255)  # Orange for medium confidence
                                        thickness = 2
                                    else:
                                        color = (0, 0, 255)  # Red for low confidence
                                        thickness = 1
                                        
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                                    
                                    # Add percentage text with background for better visibility
                                    text = f"{similarity*100:.1f}%"
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 0.6
                                    font_thickness = 2
                                    
                                    # Get text size
                                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                                    
                                    # Draw background rectangle
                                    cv2.rectangle(frame, (x, y - text_height - 5), (x + text_width, y), color, -1)
                                    
                                    # Draw text
                                    cv2.putText(frame, text, (x, y - 5),
                                              font, font_scale, (255, 255, 255), font_thickness)
                        
                        # Display match statistics on frame
                        if self.debug_mode and all_matches:
                            # Sort matches by similarity
                            all_matches.sort(key=lambda x: x[1], reverse=True)
                            
                            # Display top matches and highest seen
                            y_pos = 30
                            cv2.putText(frame, "Match Percentages:", (10, y_pos),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # Show current top 3 matches
                            for idx, (box, similarity) in enumerate(all_matches[:3], 1):
                                if similarity > self.show_threshold:
                                    y_pos += 25
                                    color = (0, 255, 0) if similarity > self.reid_threshold else \
                                           (0, 165, 255) if similarity > 0.8 else (0, 0, 255)
                                    cv2.putText(frame, f"Match {idx}: {similarity*100:.1f}%", (10, y_pos),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            # Show highest confidence seen
                            y_pos += 35
                            cv2.putText(frame, f"Highest Match: {self.highest_confidence_seen*100:.1f}% (Cam {self.highest_confidence_camera+1})",
                                      (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
                            # Print to console
                            if best_match is not None and best_similarity > self.show_threshold:
                                print(f"\rCamera {i+1} - Best Match: {best_similarity*100:.1f}% | " + 
                                      f"Highest Ever: {self.highest_confidence_seen*100:.1f}% (Cam {self.highest_confidence_camera+1})", end="")
                        
                        # If track is lost in primary camera and we find a good match here, switch cameras
                        if self.track_lost and best_match is not None:
                            box, similarity = best_match
                            
                            # Only consider very high confidence matches
                            if similarity > self.reid_threshold:
                                # Check if this is a consistent match in the same camera
                                if i == self.last_match_camera and self.similar_boxes(box, self.last_match_box):
                                    self.consecutive_matches += 1
                                    print(f"\nHigh confidence match in camera {i+1}: {similarity*100:.1f}% - Match {self.consecutive_matches}/{self.min_consecutive_matches}")
                                else:
                                    self.consecutive_matches = 1
                                
                                self.last_match_camera = i
                                self.last_match_box = box
                                
                                # Only switch if we have consistent high confidence matches
                                if self.consecutive_matches >= self.min_consecutive_matches:
                                    print(f"\nSwitching to camera {i+1} - High confidence match: {similarity*100:.1f}%")
                                    # Switch tracking to this camera
                                    self.tracking_camera = i
                                    self.tracker = cv2.legacy.TrackerCSRT_create()
                                    success = self.tracker.init(frame, box)
                                    if success:
                                        self.track_lost = False
                                        self.lost_track_frames = 0
                                        
                                        x, y, w, h = box
                                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                                        cv2.putText(frame, f"Tracking Started ({similarity*100:.1f}%)", (x, y - 10),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display the frame
                cv2.imshow(f"Camera {i+1} - {os.path.basename(self.video_sources[i])}", frame)
            
            # Handle key presses
            key = cv2.waitKey(int(1000 / self.fps)) & 0xFF
            
            if key == ord('q'):
                logger.info("User requested quit")
                self.running = False
                break
            elif key == ord('p'):
                self.paused = not self.paused
                status = "Paused" if self.paused else "Resumed"
                logger.info(f"Playback {status}")
                print(f"\nPlayback {status}")
        
        # Clean up
        for cap in self.caps:
            cap.release()
        cv2.destroyAllWindows()
        logger.info("All tracking completed")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for object selection"""
        camera_idx = param
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.paused:
                logger.info("Please pause the video ('p' key) before selecting an object")
                return
                
            frame = None
            if self.caps[camera_idx].isOpened():
                ret, frame = self.caps[camera_idx].read()
                if not ret:
                    logger.error("Failed to read frame for object selection")
                    return
                    
            # Get YOLO detections
            detected_boxes = self.detect_person_in_frame(frame)
            if not detected_boxes:
                logger.info("No objects detected in frame")
                return
                
            # Find the closest detection to the clicked point
            closest_box = None
            min_distance = float('inf')
            max_distance = 100  # Maximum pixel distance to consider
            
            for box in detected_boxes:
                box_x, box_y, box_w, box_h = box
                # Calculate center of the box
                box_center_x = box_x + box_w // 2
                box_center_y = box_y + box_h // 2
                
                # Calculate distance to clicked point
                distance = ((box_center_x - x) ** 2 + (box_center_y - y) ** 2) ** 0.5
                
                if distance < min_distance and distance < max_distance:
                    min_distance = distance
                    closest_box = box
            
            if closest_box is not None:
                # Initialize tracker with the closest detection
                try:
                    self.tracker = cv2.legacy.TrackerCSRT_create()
                    success = self.tracker.init(frame, closest_box)
                    
                    if success:
                        self.tracking_enabled = True
                        self.tracking_camera = camera_idx
                        self.track_lost = False
                        self.lost_track_frames = 0
                        
                        # Extract initial ReID features
                        self.tracked_features = self.extract_reid_features(frame, closest_box)
                        
                        logger.info(f"Started tracking object in camera {camera_idx + 1}")
                        print(f"\nTracking started in camera {camera_idx + 1}")
                        
                        # Draw the selected box
                        x, y, w, h = closest_box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, "Selected for Tracking", (x, y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.imshow(f"Camera {camera_idx+1} - {os.path.basename(self.video_sources[camera_idx])}", frame)
                    else:
                        logger.error("Failed to initialize tracker")
                        print("\nFailed to initialize tracker. Please try selecting a different object.")
                except Exception as e:
                    logger.error(f"Error initializing tracker: {str(e)}")
                    print("\nError initializing tracker. Please try again.")
            else:
                logger.info("No objects found near clicked point")
                print("\nNo objects found near clicked point. Please click closer to an object.")

    def similar_boxes(self, box1, box2, iou_threshold=0.5):
        """Check if two boxes are similar based on IoU"""
        if box1 is None or box2 is None:
            return False
            
        # Extract coordinates
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
            
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        iou = intersection / float(area1 + area2 - intersection)
        
        return iou > iou_threshold

def check_video_file(file_path):
    """Check if a video file exists and is readable"""
    if not os.path.exists(file_path):
        logger.error(f"Video file does not exist: {file_path}")
        return False
        
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {file_path}")
            return False
            
        # Try to read a frame
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.error(f"Could not read frames from video file: {file_path}")
            return False
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        logger.info(f"Video file check passed: {file_path}")
        logger.info(f"Video properties: {frame_count} frames, {width}x{height}, {fps} FPS")
        
        cap.release()
        return True
    except Exception as e:
        logger.error(f"Error checking video file {file_path}: {str(e)}")
        return False

if __name__ == "__main__":
    # List of video sources to track
    video_sources = [
        "E:\\cam2.mp4",
        "E:\\cam3.mp4"
    ]
    
    # Check if video files exist
    print("\n=== Multi-Camera Person Tracking System ===")
    print("Checking video files...")
    
    valid_sources = []
    for source in video_sources:
        if os.path.exists(source):
            print(f"Found video file: {source}")
            valid_sources.append(source)
        else:
            print(f"ERROR: Video file not found: {source}")
    
    if not valid_sources:
        print("\nERROR: No valid video sources found. Please check the file paths in the code.")
        print("Current video sources:")
        for source in video_sources:
            print(f"  - {source}")
        print("\nPlease edit the 'video_sources' list in the code with correct file paths.")
        sys.exit(1)
    
    # Test video files
    print("\nTesting video files...")
    for source in valid_sources:
        print(f"\nTesting {source}:")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"ERROR: Failed to open {source}")
            continue
            
        # Try to read first frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"ERROR: Failed to read frame from {source}")
            cap.release()
            continue
            
        # Display video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video properties:")
        print(f"- Resolution: {width}x{height}")
        print(f"- FPS: {fps}")
        
        # Show first frame
        window_name = f"Test - {os.path.basename(source)}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, frame)
        cv2.waitKey(1000)  # Show for 1 second
        cv2.destroyWindow(window_name)
        
        cap.release()
        print(f"Successfully tested {source}")
    
    print("\nPress Enter to continue with tracking...")
    input()
    
    # Check video files before starting
    final_valid_sources = []
    for source in valid_sources:
        if check_video_file(source):
            final_valid_sources.append(source)
    
    if not final_valid_sources:
        logger.error("No valid video sources found. Exiting.")
        print("\nERROR: No valid video sources found after testing. Exiting.")
        sys.exit(1)
    
    logger.info(f"Starting tracking with {len(final_valid_sources)} valid video sources")
    print(f"\nStarting tracking with {len(final_valid_sources)} valid video sources")
    
    # Create tracker and start tracking
    try:
        tracker = DrawTracker(
            final_valid_sources,
            yolo_model="yolov8n.pt",  # Using smaller model for better performance
            yolo_conf=0.5
        )
        
        # Print detailed instructions
        print("\n=== HOW TO USE THE TRACKING SYSTEM ===")
        print("1. When the videos start playing, press 'p' to pause the video")
        print("2. Click near the object you want to track")
        print("3. Press 'p' again to resume tracking")
        print("4. The system will start tracking that person/object across all cameras")
        print("\nControls:")
        print("  'p' - Pause/Resume video playback")
        print("  Click - Select object to track (video must be paused)")
        print("  'q' - Quit the application")
        print("\nPress Enter to start tracking...")
        input()
        
        tracker.run()
    except Exception as e:
        print(f"\nERROR: An error occurred during tracking: {str(e)}")
        logger.error(f"Error during tracking: {str(e)}")
        sys.exit(1) 