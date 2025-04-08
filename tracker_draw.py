import cv2
import numpy as np
import logging
import os
import sys
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tracker_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ObjectTracker")

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
    
    def run(self):
        """Run the tracker on multiple video streams."""
        self.running = True
        
        # Print instructions
        print("\n=== Multi-Camera Person Tracking System ===")
        print("Controls:")
        print("  'p' - Pause/Resume video playback")
        print("  's' - Start drawing tracking box on any camera")
        print("  'q' - Quit the application")
        print("\nVideos are playing. Press 's' to start drawing a tracking box on any camera.")
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
                if self.tracking_enabled and i == self.tracking_camera and self.tracker is not None:
                    # Update tracker
                    success, bbox = self.tracker.update(frame)
                    
                    if success:
                        # Draw tracking box
                        (x, y, w, h) = [int(v) for v in bbox]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, "Tracking", (x, y - 10), 
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
            elif key == ord('s'):
                # Pause all videos first
                was_paused = self.paused
                self.paused = True
                
                print("\n=== Draw Tracking Box ===")
                print("Click and drag on any camera window to draw a box around the person/object you want to track.")
                print("Press ENTER or SPACE to confirm your selection.")
                print("Press ESC to cancel.")
                
                # Wait for the user to draw a box
                while True:
                    # Show the current frames with the box being drawn
                    for i, frame in enumerate(frames):
                        if frame is not None:
                            frame_with_box = frame.copy()
                            
                            # Draw the box if we have start and end points and it's for this camera
                            if self.current_camera == i and self.start_point is not None and self.end_point is not None:
                                cv2.rectangle(frame_with_box, self.start_point, self.end_point, (0, 255, 0), 2)
                            
                            cv2.imshow(f"Camera {i+1} - {os.path.basename(self.video_sources[i])}", frame_with_box)
                    
                    # Wait for key press
                    key = cv2.waitKey(100) & 0xFF
                    if key == 27:  # ESC
                        print("Selection cancelled")
                        self.start_point = None
                        self.end_point = None
                        self.current_camera = None
                        break
                    elif key in [13, 32]:  # ENTER or SPACE
                        if self.start_point is not None and self.end_point is not None and self.current_camera is not None:
                            print("Box selected")
                            break
                        else:
                            print("No box drawn. Click and drag on any camera to draw a box.")
                
                # If we have a valid box, initialize the tracker
                if self.start_point is not None and self.end_point is not None and self.current_camera is not None:
                    # Convert points to bbox format (x, y, width, height)
                    x1, y1 = self.start_point
                    x2, y2 = self.end_point
                    x = min(x1, x2)
                    y = min(y1, y2)
                    w = abs(x2 - x1)
                    h = abs(y2 - y1)
                    
                    # Validate bounding box coordinates
                    frame_height, frame_width = frames[self.current_camera].shape[:2]
                    
                    # Ensure coordinates are within frame boundaries
                    x = max(0, min(x, frame_width - 1))
                    y = max(0, min(y, frame_height - 1))
                    w = max(1, min(w, frame_width - x))
                    h = max(1, min(h, frame_height - y))
                    
                    # Debug information
                    logger.info(f"Frame dimensions: {frame_width}x{frame_height}")
                    logger.info(f"Bounding box: x={x}, y={y}, w={w}, h={h}")
                    
                    # Initialize tracker only for the selected camera
                    try:
                        # Create tracker using legacy API which is more widely supported
                        tracker_types = [
                            ('CSRT', 'csrt'),
                            ('KCF', 'kcf'),
                            ('MIL', 'mil'),
                            ('Boosting', 'boosting')
                        ]
                        
                        for tracker_name, tracker_type in tracker_types:
                            try:
                                logger.info(f"Trying {tracker_name} tracker...")
                                self.tracker = cv2.legacy.TrackerCSRT_create() if tracker_type == 'csrt' else \
                                             cv2.legacy.TrackerKCF_create() if tracker_type == 'kcf' else \
                                             cv2.legacy.TrackerMIL_create() if tracker_type == 'mil' else \
                                             cv2.legacy.TrackerBoosting_create()
                                
                                frame_copy = frames[self.current_camera].copy()
                                success = self.tracker.init(frame_copy, (x, y, w, h))
                                
                                if success:
                                    self.tracking_enabled = True
                                    self.tracking_camera = self.current_camera
                                    logger.info(f"Successfully initialized {tracker_name} tracker for camera {self.current_camera+1}")
                                    print(f"Tracking started with {tracker_name} tracker for camera {self.current_camera+1}")
                                    break
                                else:
                                    logger.warning(f"{tracker_name} tracker initialization failed")
                            except Exception as e:
                                logger.warning(f"Error with {tracker_name} tracker: {str(e)}")
                                continue
                        
                        if not self.tracking_enabled:
                            logger.error("All tracker types failed to initialize")
                            print("Failed to initialize any tracker. Please try drawing a larger box or selecting a different region.")
                            self.tracker = None
                            
                    except Exception as e:
                        logger.error(f"Error during tracker initialization: {str(e)}")
                        print("Error initializing tracker. Please try again with a different region.")
                        self.tracker = None
                
                # Reset drawing state
                self.drawing = False
                self.start_point = None
                self.end_point = None
                self.current_camera = None
                
                # Restore previous pause state
                self.paused = was_paused
                
                print("\nPress 'p' to pause/resume, 's' to draw a new tracking box, or 'q' to quit")
        
        # Clean up
        for cap in self.caps:
            cap.release()
        cv2.destroyAllWindows()
        logger.info("All tracking completed")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes."""
        camera_idx = param
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            self.current_camera = camera_idx
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.current_camera == camera_idx:
                self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.current_camera == camera_idx:
                self.drawing = False
                self.end_point = (x, y)

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
        "E:\\cam1.mp4",
        "E:\\Single1.mp4"
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
        print("1. When the videos start playing, press 's' to start drawing a tracking box on any camera")
        print("2. You'll be asked to select which camera to track from (enter the number)")
        print("3. Click and drag on the video to draw a box around the person/object you want to track")
        print("4. Press ENTER or SPACE to confirm your selection")
        print("5. The system will start tracking that person/object across all cameras")
        print("\nControls:")
        print("  'p' - Pause/Resume video playback")
        print("  's' - Draw a new tracking box on a different camera")
        print("  'q' - Quit the application")
        print("\nPress Enter to start tracking...")
        input()
        
        tracker.run()
    except Exception as e:
        print(f"\nERROR: An error occurred during tracking: {str(e)}")
        logger.error(f"Error during tracking: {str(e)}")
        sys.exit(1) 