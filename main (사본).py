import sys
sys.path.insert(0, "/usr/lib/python3/dist-packages/") 

import os
import time
import cv2
import numpy as np
import serial
import requests

from ultralytics import YOLO
from tracker import Sort
from logger import Logger
from line_zone import LineZoneManager

# 설정
DAY_MODEL_PATH = "yolov8n.engine"
NIGHT_MODEL_PATH = "infrared_yolo.engine"
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

model = YOLO(DAY_MODEL_PATH)
current_model_type = "day"

tracker = Sort()
line_zone_manager = LineZoneManager()
logger = Logger()

CLASS_NAMES = {
    0: "Person",
    1: "Car",
    2: "Bus",
    3: "Truck"
}

state = 0
prev_state = -1
vehicle_memory_y = dict()
person_memory_y = dict()
vehicle_memory_point = dict()
person_memory_point = dict()

gst_input = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw, width=640, height=480, framerate=30/1 ! "
    "videoconvert ! appsink"
)
gst_output = (
    "appsrc ! videoconvert ! "
    "x264enc tune=zerolatency bitrate=1000 speed-preset=ultrafast ! "
    "rtph264pay config-interval=1 pt=96 ! "
    "udpsink host=192.168.10.101 port=8554"
)

cap = cv2.VideoCapture(gst_input, cv2.CAP_GSTREAMER)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
out = cv2.VideoWriter(gst_output, cv2.CAP_GSTREAMER, 0, 15, (640, 480), True)

def report_state_to_flask(new_state):
    try:
        url = "http://localhost:5050/api/state/update"
        data = {
            "device": "LED 1",
            "state": "on" if new_state > 0 else "off",
            "brightness": 3 if new_state > 0 else 0
        }
        response = requests.post(url, json=data, timeout=1)
        if response.status_code == 200:
            print(f"[Flask OK] 서버에 상태 전달 성공: {data}")
        else:
            print(f"[Flask WARN] 서버 응답 코드: {response.status_code} 데이터: {data}")
    except Exception as e:
        print(f"[Flask ERROR] 서버 전송 실패: {e}")

def is_night_time(frame):
    return frame.mean() < 50

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if is_night_time(frame):
        if current_model_type != "night":
            model = YOLO(NIGHT_MODEL_PATH)
            current_model_type = "night"
    else:
        if current_model_type != "day":
            model = YOLO(DAY_MODEL_PATH)
            current_model_type = "day"

    results = model.predict(frame)
    detections, classes = [], []

    if results and results[0].boxes is not None:
        detections = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

    valid_detections = []
    valid_classes = []
    center_coords = []

    for det, cls_id in zip(detections, classes):
        x1, y1, x2, y2 = map(int, det[:4])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if cls_id == 0:
            if (line_zone_manager.PERSON_X1 < cx < line_zone_manager.PERSON_X2) and (cy > line_zone_manager.PERSON_LINE_Y):
                valid_detections.append(det)
                valid_classes.append(cls_id)
                center_coords.append((cx, cy))

        elif cls_id in [1, 2, 3]:
            if cv2.pointPolygonTest(line_zone_manager.POLYGON, (cx, cy), False) >= 0:
                valid_detections.append(det)
                valid_classes.append(cls_id)
                center_coords.append((cx, cy))

    if len(valid_detections) == 0:
        tracked_objects = tracker.update(np.array([]))
        line_zone_manager.person_activated.clear()
        line_zone_manager.vehicle_activated.clear()
        state = 0
        if state != prev_state:
            report_state_to_flask(state)
            prev_state = state
        line_zone_manager.draw(frame, state)
            
    else:
        tracked_objects = tracker.update(np.array(valid_detections))
        current_active_ids = set()

        for obj, cls_id, (cx, cy) in zip(tracked_objects, valid_classes, center_coords):
            track_id = int(obj[4])
            x1, y1, x2, y2 = map(int, obj[:4])
            class_id = int(cls_id)
            class_name = CLASS_NAMES.get(class_id, "Unknown")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{class_name} {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            if class_id in [1, 2, 3]:
                prev_pt = vehicle_memory_point.get(track_id)
                if prev_pt:
                    prev_x, prev_y = prev_pt
                    prev_in = cv2.pointPolygonTest(line_zone_manager.POLYGON, prev_pt, False) < 0
                    curr_in = cv2.pointPolygonTest(line_zone_manager.POLYGON, (cx, cy), False) >= 0
                    if prev_in and curr_in and cy > prev_y:
                        if track_id not in line_zone_manager.vehicle_crossed:
                            line_zone_manager.vehicle_crossed.add(track_id)
                            logger.log_vehicle(track_id, frame.copy(), class_name, speed=0, state=state, line_drawer=line_zone_manager)
                            print(f"[Vehicle Entry] ID={track_id}, cx={cx}, cy={cy}")
                        line_zone_manager.vehicle_activated.add(track_id)
                        current_active_ids.add(track_id)
                vehicle_memory_point[track_id] = (cx, cy)

            elif class_id == 0:
                prev_y = person_memory_y.get(track_id)
                if prev_y is not None and prev_y <= line_zone_manager.PERSON_LINE_Y and cy > line_zone_manager.PERSON_LINE_Y:
                    if track_id not in line_zone_manager.person_crossed:
                        line_zone_manager.person_crossed.add(track_id)
                        logger.log_person(track_id, frame.copy(), class_name, speed=0, state=state, line_drawer=line_zone_manager)
                        print(f"[Person Crossing] ID={track_id}, cx={cx}, cy={cy}")
                person_memory_y[track_id] = cy
                line_zone_manager.person_activated.add(track_id)
                current_active_ids.add(track_id)
                
            if y2 >= FRAME_HEIGHT:
                line_zone_manager.person_activated.discard(track_id)
                line_zone_manager.vehicle_activated.discard(track_id)
            else:
                if class_id == 0:
                    line_zone_manager.person_activated.add(track_id)
                elif class_id in [1, 2, 3]:
                    line_zone_manager.vehicle_activated.add(track_id)

        if not line_zone_manager.person_activated and not line_zone_manager.vehicle_activated:
            state = 0
        elif line_zone_manager.person_activated and line_zone_manager.vehicle_activated:
            state = 3
        elif line_zone_manager.person_activated:
            state = 1
        elif line_zone_manager.vehicle_activated:
            state = 2

        if state != prev_state:
            report_state_to_flask(state)
            prev_state = state

        line_zone_manager.draw(frame, state)
    out.write(frame)
    cv2.imshow("YOLO Tracking", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
