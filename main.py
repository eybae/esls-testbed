from utils.person_handler import handle_persons
from utils.vehicle_handler import handle_vehicles
from utils.brightspot_handler import handle_brightspots
from utils.logger import Logger
from utils.tracker import Sort
from utils.line_zone import LineZoneManager

import sys
sys.path.insert(0, "/usr/lib/python3/dist-packages/")
import os
import time
import cv2
import numpy as np
import serial
import requests
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path

DAY_MODEL_PATH = "yolov8n.engine"
NIGHT_MODEL_PATH = "best.engine"

tracker = Sort()
line_zone_manager = LineZoneManager()
logger = Logger()

VEHICLE_CLASS_IDS = [2, 3, 5, 7]
CAPTURE_INTERVAL = 2.0
TURN_OFF_TIME = 10
state, prev_state = 0, -1
last_active_time = time.time()

vehicle_logged_time = {}
person_logged_time = {}
person_inside_memory = {}
bright_logged_time = {}
bright_recent_y = {}
vehicle_memory_point = {}
person_recent_y = {} 

recorder_info = {'writer': None, 'filename': None}

def is_night_time():
    now = datetime.now().time()
    return now >= datetime.strptime("20:00", "%H:%M").time() or now <= datetime.strptime("06:00", "%H:%M").time()

def start_video_recording(frame, recorder_info):
    now = datetime.now()
    hour, minute = now.hour, now.minute
    if 20 <= hour < 24:
        Path("./video").mkdir(parents=True, exist_ok=True)
        minute_block = (minute // 10) * 10
        filename = f"./video/{now.strftime('%Y%m%d_%H')}{minute_block:02d}.mp4"
        if recorder_info['writer'] is None or recorder_info['filename'] != filename:
            if recorder_info['writer']:
                recorder_info['writer'].release()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(filename, fourcc, 15.0, (640, 480))
            if not writer.isOpened():
                print(f"[ERROR] VideoWriter 생성 실패: {filename}")
                return
            recorder_info['writer'] = writer
            recorder_info['filename'] = filename
            print(f"[녹화 시작] {filename}")
        recorder_info['writer'].write(frame)
    else:
        if recorder_info['writer']:
            recorder_info['writer'].release()
            print(f"[녹화 종료] {recorder_info['filename']}")
            recorder_info['writer'] = None
            recorder_info['filename'] = None

def report_state_to_flask(new_state):
    try:
        url = "http://localhost:5050/api/state/update"
        data = {
            "device": "LED 1",
            "state": "on" if new_state > 0 else "off",
            "brightness": 3 if new_state > 0 else 0
        }
        response = requests.post(url, json=data, timeout=1)
        print(f"[Flask] 상태 전송: {data}, 응답코드: {response.status_code}")
    except Exception as e:
        print(f"[Flask ERROR] 상태 전송 실패: {e}")

gst_input = (
    "v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, framerate=30/1 ! "
    "videoconvert ! appsink drop=true sync=false"
)
gst_output = (
    "appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=1000 speed-preset=ultrafast ! "
    "rtph264pay config-interval=1 pt=96 ! udpsink host=192.168.10.101 port=8554"
)

cap = cv2.VideoCapture(gst_input, cv2.CAP_GSTREAMER)
out = cv2.VideoWriter(gst_output, cv2.CAP_GSTREAMER, 0, 30, (640, 480), True)

model_day = YOLO(DAY_MODEL_PATH, task="detect")
model_bright = YOLO(NIGHT_MODEL_PATH, task="detect")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    raw_frame = frame.copy()
    night = is_night_time()

    results = model_day.predict(frame, conf=0.25)
    detections, classes, center_coords = [], [], []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if 380 < cx < 400 and 70 < cy < 85:
                continue
            if cls_id == 0 and (line_zone_manager.PERSON_X1 < cx < line_zone_manager.PERSON_X2) and cy > line_zone_manager.PERSON_LINE_Y:
                detections.append([x1, y1, x2, y2])
                classes.append(cls_id)
                center_coords.append((cx, cy))
            elif cls_id in VEHICLE_CLASS_IDS and cv2.pointPolygonTest(line_zone_manager.POLYGON, (cx, cy), False) >= 0:
                detections.append([x1, y1, x2, y2])
                classes.append(cls_id)
                center_coords.append((cx, cy))

    tracked_objects = tracker.update(np.array(detections)) if detections else []
    person_detected = handle_persons(tracked_objects, classes, center_coords, frame, logger,
                                 line_zone_manager, person_logged_time, person_inside_memory,
                                 CAPTURE_INTERVAL, person_recent_y)
    vehicle_detected = handle_vehicles(tracked_objects, classes, center_coords, frame, logger,
                                       line_zone_manager, vehicle_logged_time, vehicle_memory_point, CAPTURE_INTERVAL)
    bright_spot_detected = handle_brightspots(night, model_bright, frame, line_zone_manager,
                                              bright_logged_time, CAPTURE_INTERVAL, logger, bright_recent_y)

    # 상태 계산 및 전이
    if len(line_zone_manager.person_activated) > 0 and (len(line_zone_manager.vehicle_activated) > 0 or bright_spot_detected):
        state = 3
        last_active_time = time.time()
    elif len(line_zone_manager.person_activated) > 0:
        state = 1
        last_active_time = time.time()
    elif len(line_zone_manager.vehicle_activated) > 0 or bright_spot_detected:
        state = 2
        last_active_time = time.time()
    elif state in [1, 2, 3]:
        state = 4  # 아무것도 감지되지 않음, 감지유지 종료 대기 시작
        last_active_time = time.time()
    elif state == 4 and (time.time() - last_active_time > TURN_OFF_TIME):
        state = 0  # 감지 종료 후 10초 이상 경과

    if state != prev_state:
        report_state_to_flask(state)
        prev_state = state

    line_zone_manager.draw(frame, state)
    out.write(frame)
    start_video_recording(raw_frame, recorder_info)

cap.release()
out.release()
cv2.destroyAllWindows()