# CAM — Real-Time Vehicle Surveillance with YOLOv11 & IP Cameras

An industrial-grade computer vision pipeline for live vehicle monitoring over CCTV/IP camera networks. Connects to NVR systems via RTSP, lets operators define custom L-shape or polygon Regions of Interest (ROI) on camera frames, and runs YOLOv11 detection and segmentation to track trucks and detect tyres in real time.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-purple)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)

---

## Overview

This project solves a real-world industrial problem: monitoring vehicle activity (trucks, tyres) in specific zones of a camera's field of view. Rather than running detection over the entire frame, operators interactively define L-shaped ROI zones that match the physical layout (lanes, loading bays, entry gates), and inference is restricted to those zones for accuracy and efficiency.

**Core capabilities:**
- Live RTSP stream ingestion from CP Plus NVR systems (multi-channel)
- Interactive L-shape and polygon ROI annotation tool
- YOLOv11 object detection (vehicle detection)
- YOLOv11 instance segmentation (tyre segmentation)
- REST API backend (FastAPI) for serving predictions
- JSON export of detection results and ROI configurations

---

## System Architecture

```
CP Plus NVR / IP Cameras
         │  RTSP Stream
         ▼
┌─────────────────────┐
│   RTSP Ingestion    │  ← cv2.VideoCapture (multi-channel, multi-format)
│   (OpenCV)          │
└──────────┬──────────┘
           │  Frames
           ▼
┌─────────────────────┐
│  ROI Zone Filter    │  ← L-shape / Polygon zones (saved as JSON)
│  (Custom geometry)  │    Operators click 3 points → defines detection zone
└──────────┬──────────┘
           │  Cropped/Masked Frames
           ▼
┌──────────────────────────┐
│       YOLOv11            │
│  ┌──────────┐ ┌────────┐ │
│  │Detection │ │  Seg   │ │  ← yolo11n.pt (detection)
│  │(Trucks)  │ │(Tyres) │ │  ← yolo11n-seg.pt (segmentation)
│  └──────────┘ └────────┘ │
└──────────┬───────────────┘
           │  Bounding Boxes / Masks
           ▼
┌─────────────────────┐
│    FastAPI Backend  │  ← REST endpoint for predictions
│    (tfastapi/)      │
└─────────────────────┘
           │
           ▼
     JSON Output / Live Display
```

---

## Key Features

- **Multi-format RTSP support** — automatically tests 6 CP Plus URL formats to find the working connection
- **Interactive L-shape ROI tool** — click 3 points (A, B, C) on any camera frame to define a detection zone; zones saved as JSON for reuse
- **Polygon ROI support** — for non-L-shape zones and complex camera angles
- **Dual YOLO models** — `yolo11n.pt` for fast vehicle detection, `yolo11n-seg.pt` for tyre segmentation
- **Truck detection & tracking** — specialized notebook for heavy vehicle monitoring
- **Batch ROI definition** — define zones for multiple camera frames in one session
- **Coordinate system** — bottom-left origin (math coordinates) for intuitive zone specification
- **FastAPI backend** — serves detection results over HTTP for integration with dashboards or other services

---

## Project Structure

```
CAM/
├── final.ipynb              # L-shape ROI definition tool (interactive, batch)
├── truck.ipynb              # Truck detection and tracking pipeline
├── file.ipynb               # Experimental / development notebook
├── 05-07-25.ipynb           # Dataset / training experiments
├── test_nvr.py              # Single-channel NVR RTSP connection tester
├── test_nvr_multiple.py     # Multi-format RTSP URL tester (6 formats)
├── yolo11n.pt               # YOLOv11 nano detection model
├── yolo11n-seg.pt           # YOLOv11 nano segmentation model
├── lshape_*.json            # Saved L-shape ROI configs per camera frame
├── polygon_ipcam.json       # Saved polygon ROI config for IP cam
├── plain.jpeg               # Sample camera frame
├── model_training/          # Model training scripts and configs
├── output_json/             # Detection output JSONs
├── runs/                    # YOLO training run artifacts
├── test_images/             # Sample test frames
├── tfastapi/                # FastAPI prediction server
├── tyre_seg/                # Tyre segmentation pipeline
├── yolo_1/ yolo_2/          # YOLO experiment variants
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- CP Plus NVR (or any RTSP-compatible IP camera)
- Camera accessible on the same network

### Install Dependencies

```bash
pip install ultralytics opencv-python numpy fastapi uvicorn
```

### 1. Test Your NVR Connection

```bash
# Single camera, single format
python test_nvr.py

# Auto-detect working RTSP URL format (tries 6 formats)
python test_nvr_multiple.py
```

Edit the `IP`, `USERNAME`, `PASSWORD`, and `CHANNEL` at the top of either script. The multi-format tester will print the working RTSP URL to use in downstream scripts.

### 2. Define ROI Zones (Interactive)

Run `final.ipynb` in Jupyter. For each camera frame in `test_images/`:

1. A window opens showing the camera frame with coordinate axes
2. **Click 3 points** (A → B → C) to define the L-shape zone
3. Press `s` to save, `r` to reset, `q` to skip
4. Zone is saved to `output_json/<frame>_lshape.json`

```
L-Shape Zone Definition:
  A ──── B
         │
         C
```

The tool automatically computes line equations for AB and BC, identifies 4 spatial regions relative to these lines, and saves all geometry to JSON.

### 3. Run Vehicle Detection

Open `truck.ipynb` to run YOLOv11 detection on RTSP stream or test images.

### 4. Start FastAPI Server

```bash
cd tfastapi
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## ROI Zone Format (JSON)

Each saved zone JSON contains the 3 clicked points and derived geometry:

```json
{
  "A": [320, 480],
  "B": [640, 480],
  "C": [640, 200],
  "line_AB": "y = 0.00x + 480.00",
  "line_BC": "x = 640",
  "S": [0, 200],
  "R": [1280, 200]
}
```

This is loaded at inference time to mask detections outside the defined zone.

---

## RTSP URL Formats Supported

The connection tester automatically tries all standard CP Plus NVR formats:

| Format | URL Pattern |
|--------|------------|
| Sub stream (recommended) | `rtsp://user:pass@IP:554/cam/realmonitor?channel=N&subtype=1` |
| Main stream | `rtsp://user:pass@IP:554/cam/realmonitor?channel=N&subtype=0` |
| Channel format | `rtsp://user:pass@IP:554/ch0N/0` |
| Alt port | `rtsp://user:pass@IP:8554/cam/realmonitor?channel=N&subtype=1` |
| Legacy | `rtsp://user:pass@IP:554/1N` |
| Generic | `rtsp://user:pass@IP:554/Streaming/Channels/N01` |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv11 (Ultralytics) |
| Instance Segmentation | YOLOv11-seg |
| Video Streaming | OpenCV (RTSP) |
| ROI Annotation | OpenCV + NumPy (custom geometry) |
| REST API | FastAPI + Uvicorn |
| Notebooks | Jupyter |

---

## License

MIT License
