# ESL Testbed: Real-Time IR Object Detection

This project uses YOLO-based models (daylight and infrared) to detect vehicles and people in real-time,
with automatic model switching, logging, and GStreamer-based streaming.

## Features
- Day/Night auto switching using brightness or time
- Vehicle/person detection with zone filtering
- Tracker-based ID assignment
- Automatic log & capture system
- 48h auto-cleanup
- Real-time UDP video output via GStreamer

## Run
```bash
python3 main.py
