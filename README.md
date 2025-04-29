# README.md

# ESL Testbed: Real-Time IR Object Detection

This project implements real-time object detection using YOLO models with dual model switching (daylight and infrared), including tracking, zone-based filtering, logging, and GStreamer-based streaming output.

## 🔧 Features
- Daytime and infrared model switching based on brightness
- Person and vehicle detection with zone constraints
- SORT tracker for consistent ID assignment
- State reporting to Flask API
- Auto-saving logs and annotated captures
- Auto-cleanup for files older than 48 hours
- Real-time UDP video output using GStreamer

## 📦 Dependencies
- Python 3.8+
- OpenCV
- NumPy
- PyTorch
- Ultralytics (YOLO)
- Requests

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Run
```bash
python3 main.py
```

Ensure the following files are available:
- `yolov8n.engine` (day model)
- `infrared_yolo.engine` (night model)
- Line and tracking logic in `line_zone.py`, `tracker.py`

## 📁 Directory Structure
```
Dev/esls-testbed/
├── main.py
├── logger.py
├── tracker.py
├── line_zone.py
├── yolov8n.engine
├── infrared_yolo.engine
├── log/              ← logs auto-created per object & date
├── capture/          ← image captures auto-saved
```

---
