*# Multi-Camera Person Tracking System  

![System Demo](demo.gif) *(Example GIF showing multi-camera tracking)*  

A real-time person tracking system that follows individuals across multiple camera feeds using **YOLOv8 detection**, **CSRT tracking**, and **ReID (Re-Identification)** with deep learning.

---

## 🔧 Key Technologies  

| Component          | Technology Used          | Purpose                          |
|--------------------|-------------------------|----------------------------------|
| Object Detection   | YOLOv8n (`yolov8n.pt`)  | Detect persons in video frames   |
| Single-Camera Tracking | OpenCV CSRT Tracker  | High-accuracy frame-to-frame tracking |
| Cross-Camera ReID  | OSNet (`osnet_x0_msmt17.pt`) | Re-identify persons across cameras |
| Framework          | PyTorch + OpenCV        | Deep learning & computer vision  |

---

## 🚀 Features  

- **Multi-Camera Switching**: Automatically follows targets across different video feeds  
- **Tracking Recovery**: Robust re-detection after occlusions or tracking failures  
- **Confidence-Based Matching**: Uses cosine similarity (threshold: 90%) for ReID  
- **Debug Visualization**: Real-time display of tracking confidence scores  
- **Pause/Resume**: Interactive control during operation  

---

## 📦 Installation (Docker)  

### Prerequisites  

- Docker ([Install Guide](https://docs.docker.com/engine/install/))  
- NVIDIA GPU + Docker (Recommended)  

```bash

  # Verify NVIDIA setup
  nvidia-smi
```

## Build & Run

### Clone repository:

    ```
    git clone https://github.com/your-repo/multi-cam-tracking.git
    cd multi-cam-tracking
    ```

### Build Docker image:

```
docker build -t tracker .
```

### Run with GPU:

```
docker run -it --gpus all \
  -v $(pwd):/app \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  cam-tracker
```

## 🖥️ Usage

1. Prepare Video Files
Place your video files (e.g., cam1.mp4, cam2.mp4) in the project root and update video_sources in the code.

2. Controls
Key	Action
p	Pause/Resume video
Click	Select person to track (when paused)
q	Quit application
3. Tracking Process
System detects all persons using YOLOv8

Click near target person when paused

System will:

- Track in current camera with CSRT

- Search other cameras using ReID model

- Switch views automatically when match confidence >90%

## ⚙️ Configuration
Modify these parameters in tracker.py:

#### Video Sources
```
video_sources = ["cam1.mp4", "cam2.mp4"]  
```

#### Detection
```
yolo_model = "yolov8n.pt"  
yolo_conf = 0.5  # Detection confidence threshold  
```

#### ReID Settings
```

reid_threshold = 0.90  # Min similarity for camera switch  
recovery_threshold = 0.85  # Lost-track recovery threshold  
max_lost_frames = 30  # Frames to keep searching  
```

### 📊 Performance Optimization

For Better Accuracy
Increase reid_threshold (up to 0.95)

Use higher resolution videos

For Faster Processing
Lower yolo_conf (e.g., 0.3)

Reduce video resolution

Use yolov8s.pt (smaller model)

## 🐛 Troubleshooting
|Issue |	Solution|
|----------|-----------|
|"Video file not found"|	Check file paths + permissions|
|Low FPS	|Enable GPU + reduce resolution|
|X11 display errors	|Run ```xhost +local:docker```|
|Tracking jumps between persons|	Increase reid_threshold|

## 📂 File Structure

```
.
├── Dockerfile
├── README.md
├── tracker.py               # Main application
├── yolov8n.pt              # YOLOv8 nano model
├── osnet_x0_msmt17.pt      # ReID model
├── cam1.mp4                # Sample videos
└── cam2.mp4
```
