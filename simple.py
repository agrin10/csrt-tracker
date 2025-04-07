import cv2
import threading
from queue import Queue
import time
import logging
import os
import sys
import numpy as np
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

class ObjectTracker:
    def __init__(self, video_source, tracker_type="KCF", queue_size=30, skip_frames=1, resize_factor=1.0, 
                 use_yolo=False, yolo_model="yolov8m.pt", yolo_conf=0.5, yolo_classes=None, debug=False):
        self.video_source = video_source
        self.tracker_type = tracker_type
        self.queue_size = queue_size
        self.skip_frames = skip_frames  # Process every Nth frame
        self.resize_factor = resize_factor  # Resize factor (1.0 = original size)
        self.use_yolo = use_yolo
        self.yolo_model = yolo_model
        self.yolo_conf = yolo_conf
        self.yolo_classes = yolo_classes  # List of class IDs to detect (None = all)
        self.debug = debug  # Debug mode flag
        
        # Create tracker based on type
        if tracker_type == "CSRT":
            self.tracker = cv2.TrackerCSRT_create()
        elif tracker_type == "KCF":
            self.tracker = cv2.TrackerKCF_create()
        elif tracker_type == "MOSSE":
            self.tracker = cv2.TrackerMOSSE_create()
        else:
            logger.warning(f"Unknown tracker type {tracker_type}, using KCF")
            self.tracker = cv2.TrackerKCF_create()
            
        # Initialize YOLO if needed
        if self.use_yolo:
            try:
                logger.info(f"Loading YOLO model: {yolo_model}")
                self.yolo = YOLO(yolo_model)
                logger.info("YOLO model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {str(e)}")
                self.use_yolo = False
                
        self.video = cv2.VideoCapture(video_source)
        self.frame_queue = Queue(maxsize=queue_size)
        self.result_queue = Queue()
        self.running = False
        self.window_name = f"Frame - {os.path.basename(video_source)}"  # Use just the filename
        
        # Get video properties for timing
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_delay = max(int(1000 / self.fps), 1) if self.fps > 0 else 1
        self.last_frame_time = 0
        
        # Performance metrics
        self.frames_read = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.start_time = 0
        
        # Log video properties
        logger.info(f"Initialized tracker for {video_source}")
        logger.info(f"Video properties: FPS={self.fps}, Frame delay={self.frame_delay}ms")
        logger.info(f"Tracker settings: Type={tracker_type}, Queue size={queue_size}, Skip frames={skip_frames}, Resize factor={resize_factor}")
        if self.use_yolo:
            logger.info(f"YOLO settings: Model={yolo_model}, Confidence={yolo_conf}, Classes={yolo_classes}")
        
        # Check if video opened successfully
        if not self.video.isOpened():
            logger.error(f"Failed to open video source: {video_source}")
            raise ValueError(f"Could not open video source: {video_source}")
        
    def initialize_tracker(self, frame, bbox):
        try:
            # Resize frame if needed
            if self.resize_factor != 1.0:
                height, width = frame.shape[:2]
                new_width = int(width * self.resize_factor)
                new_height = int(height * self.resize_factor)
                frame = cv2.resize(frame, (new_width, new_height))
                # Adjust bbox for resized frame
                bbox = (int(bbox[0] * self.resize_factor), 
                        int(bbox[1] * self.resize_factor),
                        int(bbox[2] * self.resize_factor), 
                        int(bbox[3] * self.resize_factor))
            
            self.tracker.init(frame, bbox)
            logger.info(f"Tracker initialized with bbox: {bbox}")
        except Exception as e:
            logger.error(f"Failed to initialize tracker: {str(e)}")
            raise
        
    def detect_objects(self, frame):
        """Detect objects in the frame using YOLO"""
        if not self.use_yolo:
            return None
            
        try:
            if self.debug:
                logger.debug(f"Running YOLO detection on frame for {self.video_source}")
                
            # Run YOLO detection
            results = self.yolo.predict(frame, imgsz=640, conf=self.yolo_conf, iou=0.4)
            
            # Check if results are valid
            if not results or len(results) == 0:
                logger.warning("YOLO detection returned no results")
                return None
                
            # Get the first result
            result = results[0]
            
            if self.debug:
                logger.debug(f"YOLO detection completed for {self.video_source}")
                
            # Process results
            detections = []
            
            # Check if boxes exist
            if not hasattr(result, 'boxes') or not hasattr(result.boxes, 'data'):
                logger.warning("YOLO result has no boxes data")
                return None
                
            # Process each detection
            for det in result.boxes.data:
                try:
                    x1, y1, x2, y2, conf, cls = det.tolist()
                    
                    # Filter by class if specified
                    if self.yolo_classes is None or int(cls) in self.yolo_classes:
                        w, h = x2 - x1, y2 - y1
                        bbox = (int(x1), int(y1), int(w), int(h))
                        
                        # Get class name
                        cls_id = int(cls)
                        cls_name = result.names[cls_id] if hasattr(result, 'names') and cls_id in result.names else f"class_{cls_id}"
                        
                        detections.append({
                            'bbox': bbox,
                            'class_id': cls_id,
                            'class_name': cls_name,
                            'confidence': conf
                        })
                        
                        if self.debug:
                            logger.debug(f"Detected {cls_name} with confidence {conf:.2f} at {bbox}")
                except Exception as e:
                    logger.error(f"Error processing detection: {str(e)}")
                    continue
                    
            if self.debug:
                logger.debug(f"YOLO detected {len(detections)} objects in {self.video_source}")
            return detections
        except Exception as e:
            logger.error(f"Error in YOLO detection: {str(e)}")
            return None
        
    def frame_reader(self):
        """Thread for reading frames with timing control"""
        self.frames_read = 0
        self.start_time = time.time()
        frame_count = 0
        
        logger.info(f"Starting frame reader for {self.video_source}")
        
        while self.running:
            try:
                # Maintain original video timing
                elapsed_time = time.time() - self.last_frame_time
                if elapsed_time < 1/self.fps:
                    time.sleep(1/self.fps - elapsed_time)
                
                # Skip frames if needed
                if self.skip_frames > 1:
                    for _ in range(self.skip_frames - 1):
                        ret, _ = self.video.read()
                        if not ret:
                            self.running = False
                            break
                        self.frames_read += 1
                
                ret, frame = self.video.read()
                if not ret:
                    logger.info(f"End of video reached for {self.video_source}")
                    self.running = False
                    break
                
                self.frames_read += 1
                
                # Resize frame if needed
                if self.resize_factor != 1.0:
                    height, width = frame.shape[:2]
                    new_width = int(width * self.resize_factor)
                    new_height = int(height * self.resize_factor)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                    frame_count += 1
                else:
                    self.frames_dropped += 1
                    logger.warning(f"Frame queue full for {self.video_source}, dropping frame")
                
                self.last_frame_time = time.time()
                
                # Log performance metrics every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    fps_actual = frame_count / elapsed if elapsed > 0 else 0
                    logger.info(f"Performance: {self.video_source} - {frame_count} frames, {fps_actual:.2f} FPS, {self.frames_dropped} dropped")
                    
            except Exception as e:
                logger.error(f"Error in frame reader: {str(e)}")
                self.running = False
                break
                
        logger.info(f"Frame reader stopped for {self.video_source}, processed {frame_count} frames, dropped {self.frames_dropped} frames")
    
    def tracker_processor(self):
        processed_frames = 0
        successful_tracks = 0
        
        logger.info(f"Starting tracker processor for {self.video_source}")
        
        while self.running or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=1)
                
                # Run YOLO detection if enabled
                detections = None
                tracking_box = None
                if self.use_yolo:
                    detections = self.detect_objects(frame)
                    # Update tracker with YOLO detection if a person is found
                    if detections:
                        for det in detections:
                            if det['class_name'] == 'person':
                                bbox = det['bbox']
                                # Reinitialize tracker with YOLO detection
                                self.tracker = cv2.TrackerKCF_create()  # Create new tracker instance
                                self.tracker.init(frame, bbox)
                                tracking_box = bbox
                                successful_tracks += 1
                                break
                
                # Only use traditional tracking if no person was detected by YOLO
                if tracking_box is None:
                    success, bbox = self.tracker.update(frame)
                    if success:
                        tracking_box = bbox
                        successful_tracks += 1
                
                processed_frames += 1
                self.frames_processed += 1
                
                result = {
                    'frame': frame,
                    'success': tracking_box is not None,
                    'bbox': tracking_box,
                    'detections': detections
                }
                self.result_queue.put(result)
                
                # Log tracking performance every 100 frames
                if processed_frames % 100 == 0:
                    success_rate = (successful_tracks / processed_frames) * 100 if processed_frames > 0 else 0
                    logger.info(f"Tracking performance: {self.video_source} - {success_rate:.2f}% success rate")
                    
            except Exception as e:
                logger.error(f"Error in tracker processor: {str(e)}")
                if self.running:
                    break
                    
        logger.info(f"Tracker processor stopped for {self.video_source}, processed {processed_frames} frames")
    
    def start_tracking(self, initial_bbox):
        self.running = True
        
        logger.info(f"Starting tracking for {self.video_source} with bbox: {initial_bbox}")
        
        reader_thread = threading.Thread(target=self.frame_reader, daemon=True)
        reader_thread.start()
        
        processor_thread = threading.Thread(target=self.tracker_processor, daemon=True)
        processor_thread.start()
        
        frame_count = 0
        start_time = time.time()
        
        # Create window before entering the loop
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except Exception as e:
            logger.error(f"Failed to create window {self.window_name}: {str(e)}")
        
        while self.running:
            try:
                result = self.result_queue.get(timeout=1)
                frame = result['frame']
                frame_count += 1
                
                # Draw tracking box only if not using YOLO
                if not self.use_yolo and result['success']:
                    (x, y, w, h) = [int(i) for i in result['bbox']]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Tracking", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif not self.use_yolo:
                    cv2.putText(frame, "Tracking failure detected", (50, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                
                # Draw YOLO detections
                if result['detections']:
                    for det in result['detections']:
                        bbox = det['bbox']
                        cls_name = det['class_name']
                        conf = det['confidence']
                        
                        (x, y, w, h) = bbox
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(frame, f"{cls_name} {conf:.2f}", (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Display frame with error handling
                try:
                    cv2.imshow(self.window_name, frame)
                except Exception as e:
                    logger.error(f"Failed to display frame: {str(e)}")
                    # Try to recreate the window
                    try:
                        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                        cv2.imshow(self.window_name, frame)
                    except:
                        logger.error(f"Failed to recreate window {self.window_name}")
                
                # Use calculated frame delay based on video FPS
                key = cv2.waitKey(self.frame_delay) & 0xFF
                if key == ord('q'):
                    logger.info(f"User requested quit for {self.video_source}")
                    self.running = False
                    
            except Exception as e:
                logger.error(f"Display error: {str(e)}")
                # Don't break the loop for display errors, just continue
                continue
        
        # Calculate and log performance metrics
        elapsed = time.time() - start_time
        fps_actual = frame_count / elapsed if elapsed > 0 else 0
        logger.info(f"Display performance: {self.video_source} - {frame_count} frames, {fps_actual:.2f} FPS")
        
        self.video.release()
        
        # Check if window exists before destroying it
        try:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(self.window_name)
        except:
            logger.warning(f"Could not destroy window {self.window_name} - it may not exist")
            
        logger.info(f"Tracking stopped for {self.video_source}")

class MultiVideoTracker:
    def __init__(self, video_sources, tracker_type="KCF", queue_size=30, skip_frames=1, resize_factor=1.0,
                 use_yolo=False, yolo_model="yolov8m.pt", yolo_conf=0.5, yolo_classes=None, debug=False):
        self.trackers = []
        self.tracker_threads = []
        self.video_sources = video_sources
        self.initial_bboxes = []  # Store initial bounding boxes
        self.tracker_type = tracker_type
        self.queue_size = queue_size
        self.skip_frames = skip_frames
        self.resize_factor = resize_factor
        self.use_yolo = use_yolo
        self.yolo_model = yolo_model
        self.yolo_conf = yolo_conf
        self.yolo_classes = yolo_classes
        self.debug = debug  # Debug mode flag
        
        logger.info(f"Initializing MultiVideoTracker with {len(video_sources)} video sources")
        logger.info(f"Global settings: Tracker type={tracker_type}, Queue size={queue_size}, Skip frames={skip_frames}, Resize factor={resize_factor}")
        if self.use_yolo:
            logger.info(f"YOLO settings: Model={yolo_model}, Confidence={yolo_conf}, Classes={yolo_classes}")
        
    def initialize_trackers(self):
        for video_source in self.video_sources:
            try:
                logger.info(f"Initializing tracker for {video_source}")
                tracker = ObjectTracker(
                    video_source, 
                    tracker_type=self.tracker_type,
                    queue_size=self.queue_size,
                    skip_frames=self.skip_frames,
                    resize_factor=self.resize_factor,
                    use_yolo=self.use_yolo,
                    yolo_model=self.yolo_model,
                    yolo_conf=self.yolo_conf,
                    yolo_classes=self.yolo_classes,
                    debug=self.debug
                )
                ret, frame = tracker.video.read()
                if not ret:
                    logger.error(f"Error reading video source: {video_source}")
                    continue
                    
                bbox = cv2.selectROI(tracker.window_name, frame, False)
                cv2.destroyWindow(tracker.window_name)
                
                tracker.initialize_tracker(frame, bbox)
                self.trackers.append(tracker)
                self.initial_bboxes.append(bbox)  # Store the initial bbox
                logger.info(f"Successfully initialized tracker for {video_source}")
            except Exception as e:
                logger.error(f"Failed to initialize tracker for {video_source}: {str(e)}")
    
    def start_all_tracking(self):
        if not self.trackers:
            logger.error("No trackers initialized, cannot start tracking")
            return
            
        logger.info(f"Starting tracking for {len(self.trackers)} videos")
        
        for tracker, initial_bbox in zip(self.trackers, self.initial_bboxes):
            thread = threading.Thread(target=tracker.start_tracking, 
                                   args=(initial_bbox,),
                                   daemon=True)
            self.tracker_threads.append(thread)
            thread.start()
            logger.info(f"Started tracking thread for {tracker.video_source}")
        
        # Wait for all threads to complete
        for thread in self.tracker_threads:
            thread.join()
        
        # Clean up any remaining windows
        cv2.destroyAllWindows()
        logger.info("All tracking completed")

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
        "E:\\Double1.mp4",
        "E:\\Single1.mp4"  # Add more video sources as needed
    ]
    
    # Check video files before starting
    valid_sources = []
    for source in video_sources:
        if check_video_file(source):
            valid_sources.append(source)
    
    if not valid_sources:
        logger.error("No valid video sources found. Exiting.")
        sys.exit(1)
    
    logger.info(f"Starting tracking with {len(valid_sources)} valid video sources")
    
    # Create multi-tracker with optimized settings for YOLO
    multi_tracker = MultiVideoTracker(
        valid_sources,
        tracker_type="KCF",  # Faster than CSRT
        queue_size=60,       # Increased queue size for YOLO processing
        skip_frames=2,       # Process every 2nd frame to reduce load
        resize_factor=0.75,  # Reduce resolution to 75% for better performance
        use_yolo=True,       # Enable YOLO detection
        yolo_model="yolov8n.pt",  # Use YOLOv8 nano model (as shown in logs)
        yolo_conf=0.5,       # Confidence threshold
        yolo_classes=[0],    # Only detect people (class 0)
        debug=True           # Enable debug mode
    )
    
    multi_tracker.initialize_trackers()
    multi_tracker.start_all_tracking() 