# main.py 개선본
# main_ori.py 로직을 기반으로 bright_spot 추가 및 state 반영 오류 수정


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
from datetime import datetime, timedelta
from pathlib import Path

# 설정
VIDEO_PATH = "/dev/video0"
DAY_MODEL_PATH = "yolov8n.engine"
NIGHT_MODEL_PATH = "best.engine"
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

tracker = Sort()
line_zone_manager = LineZoneManager()
logger = Logger()

CLASS_NAMES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
TURN_OFF_TIME = 10
state = 0
prev_state = -1
last_active_time = time.time()

vehicle_memory_point = dict()
person_memory_y = dict()
bright_memory_coord = set()

def is_night_time():
    now = datetime.now().time()
    return now >= datetime.strptime("20:00", "%H:%M").time() or now <= datetime.strptime("06:00", "%H:%M").time()

def start_video_recording(frame, recorder_info):
    now = datetime.now()
    hour, minute = now.hour, now.minute
    print(f"[녹화 디버그] 현재 시각: {hour}:{minute}")

    if 20 <= hour < 24:
        Path("./video").mkdir(parents=True, exist_ok=True)  # 안전하게 반복 호출

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
    "appsrc ! videoconvert ! "
    "x264enc tune=zerolatency bitrate=1000 speed-preset=ultrafast ! "
    "rtph264pay config-interval=1 pt=96 ! "
    "udpsink host=192.168.10.101 port=8554"
)

cap = cv2.VideoCapture(gst_input, cv2.CAP_GSTREAMER)
out = cv2.VideoWriter(gst_output, cv2.CAP_GSTREAMER, 0, 30, (640, 480), True)

model_day = YOLO(DAY_MODEL_PATH)
model_bright = YOLO(NIGHT_MODEL_PATH)

recorder_info = {
    'writer': None,
    'filename': None
}


while True:
    ret, frame = cap.read()
    if not ret:
        break

    raw_frame = frame.copy()
    
    # Day/Night 모델 사용 구분
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

            if cls_id == 0 and (line_zone_manager.PERSON_X1 < cx < line_zone_manager.PERSON_X2) and (cy > line_zone_manager.PERSON_LINE_Y):
                detections.append([x1, y1, x2, y2])
                classes.append(cls_id)
                center_coords.append((cx, cy))
            elif cls_id in VEHICLE_CLASS_IDS and cv2.pointPolygonTest(line_zone_manager.POLYGON, (cx, cy), False) >= 0:
                detections.append([x1, y1, x2, y2])
                classes.append(cls_id)
                center_coords.append((cx, cy))

    # 야간 bright_spot 추가
    bright_spot_detected = False
    if night:
        results_bright = model_bright.predict(frame, conf=0.25)
        if results_bright and results_bright[0].boxes is not None:
            for box in results_bright[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls_id = int(box.cls[0])
                if cls_id != 0:
                    continue
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if cv2.pointPolygonTest(line_zone_manager.POLYGON, (cx, cy), False) >= 0:
                    coord_key = f"bright_{cx}_{cy}"
                    if coord_key not in bright_memory_coord:
                        logger.log_vehicle(-99, frame.copy(), "bright_spot", 0, 2, line_zone_manager)
                        bright_memory_coord.add(coord_key)
                    line_zone_manager.vehicle_activated.add(coord_key)
                    bright_spot_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "bright_spot", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    tracked_objects = tracker.update(np.array(detections)) if detections else []

    for obj, cls_id, (cx, cy) in zip(tracked_objects, classes, center_coords):
        track_id = int(obj[4])
        x1, y1, x2, y2 = map(int, obj[:4])
        class_name = CLASS_NAMES.get(cls_id, "Unknown")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        if cls_id in VEHICLE_CLASS_IDS:
            prev_pt = vehicle_memory_point.get(track_id, (cx, cy - 1))
            was_outside = cv2.pointPolygonTest(line_zone_manager.POLYGON, prev_pt, False) < 0
            is_inside = cv2.pointPolygonTest(line_zone_manager.POLYGON, (cx, cy), False) >= 0
            downward = cy > prev_pt[1]
            if was_outside and is_inside and downward:
                if track_id not in line_zone_manager.vehicle_crossed:
                    line_zone_manager.vehicle_crossed.add(track_id)
                    logger.log_vehicle(track_id, frame.copy(), class_name, 0, 2, line_zone_manager)
            if is_inside:
                line_zone_manager.vehicle_activated.add(track_id)
            vehicle_memory_point[track_id] = (cx, cy)

        elif cls_id == 0:
            prev_y = person_memory_y.get(track_id, cy - 5)
            if (145 < prev_y < 150) and cy >= 150:
                if track_id not in line_zone_manager.person_crossed:
                    line_zone_manager.person_crossed.add(track_id)
                    logger.log_person(track_id, frame.copy(), "Person", 0, 1, line_zone_manager)
            if (line_zone_manager.PERSON_X1 < cx < line_zone_manager.PERSON_X2) and (cy >= line_zone_manager.PERSON_LINE_Y):
                line_zone_manager.person_activated.add(track_id)
            person_memory_y[track_id] = cy
            
    active_person_ids = {int(obj[4]) for obj, cls_id in zip(tracked_objects, classes) if int(cls_id) == 0}
    active_vehicle_ids = {int(obj[4]) for obj, cls_id in zip(tracked_objects, classes) if int(cls_id) in VEHICLE_CLASS_IDS}

    line_zone_manager.person_activated.intersection_update(active_person_ids)
    line_zone_manager.vehicle_activated.intersection_update(active_vehicle_ids)

    # State 판단
    if not line_zone_manager.person_activated and not line_zone_manager.vehicle_activated and not bright_spot_detected:
        if state in [1, 2, 3]:
            state = 4
            last_active_time = time.time()
        elif state == 4 and time.time() - last_active_time > TURN_OFF_TIME:
            state = 0
    else:
        if line_zone_manager.person_activated and (line_zone_manager.vehicle_activated or bright_spot_detected):
            state = 3
        elif line_zone_manager.person_activated:
            state = 1
        elif line_zone_manager.vehicle_activated or bright_spot_detected:
            state = 2

    if state != prev_state:
        report_state_to_flask(state)
        prev_state = state

    line_zone_manager.draw(frame, state)
    out.write(frame)
    
    start_video_recording(raw_frame, recorder_info)
    #cv2.imshow("YOLO Tracking", frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

cap.release()
out.release()
cv2.destroyAllWindows()
