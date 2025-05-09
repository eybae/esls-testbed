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
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck
TURN_OFF_TIME = 10
state = 0
prev_state = -1
last_active_time = time.time()

vehicle_memory_point = dict()
person_memory_y = dict()

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

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 수동 모드
cap.set(cv2.CAP_PROP_EXPOSURE, -6)         # 노출 값은 장비 성능에 맞게 조정
cap.set(cv2.CAP_PROP_FPS, 15)              # FPS 고정

out = cv2.VideoWriter(gst_output, cv2.CAP_GSTREAMER, 0, 15, (640, 480), True)

model = YOLO(DAY_MODEL_PATH)

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

while True:
    ret, frame = cap.read()
    if not ret:
        break   
    results = model.predict(frame, conf=0.25)  # 신뢰도 threshold 낮춰 감지 민감도 향상
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
        
        # 특정 좌표에 대한 오검출 필터링 (파란 사각형 위치)
        if 380 < cx < 400 and 70 < cy < 85:
            #print(f"[Filtered] False detection at ({cx}, {cy})")
            continue

        if cls_id == 0:
            if (line_zone_manager.PERSON_X1 < cx < line_zone_manager.PERSON_X2) and (cy > line_zone_manager.PERSON_LINE_Y):
                valid_detections.append(det)
                valid_classes.append(cls_id)
                center_coords.append((cx, cy))

        elif cls_id in VEHICLE_CLASS_IDS:
            if cv2.pointPolygonTest(line_zone_manager.POLYGON, (cx, cy), False) >= 0:
                valid_detections.append(det)
                valid_classes.append(cls_id)
                center_coords.append((cx, cy))

    if len(valid_detections) == 0:
        tracked_objects = tracker.update(np.array([]))
        line_zone_manager.person_activated.clear()
        line_zone_manager.vehicle_activated.clear()

        if state in [1, 2, 3]:
            state = 4
            last_active_time = time.time()
        elif state == 4 and time.time() - last_active_time > TURN_OFF_TIME:
            state = 0

        if state != prev_state:
            report_state_to_flask(state)
            prev_state = state

        line_zone_manager.draw(frame, state)
        out.write(frame)
        cv2.imshow("YOLO Tracking", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        continue

    tracked_objects = tracker.update(np.array(valid_detections))

    for obj, cls_id, (cx, cy) in zip(tracked_objects, valid_classes, center_coords):
        track_id = int(obj[4])
        x1, y1, x2, y2 = map(int, obj[:4])
        class_id = int(cls_id)
        class_name = CLASS_NAMES.get(class_id, "Unknown")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{class_name}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        if class_id in VEHICLE_CLASS_IDS:
            prev_pt = vehicle_memory_point.get(track_id)
            if prev_pt:
                prev_x, prev_y = prev_pt
            else:
                prev_x, prev_y = cx, cy - 1
                #print(f"[Debug] New vehicle track_id={track_id}, forcing downward detection")

            was_outside = cv2.pointPolygonTest(line_zone_manager.POLYGON, (prev_x, prev_y), False) < 0
            is_inside = cv2.pointPolygonTest(line_zone_manager.POLYGON, (cx, cy), False) >= 0
            downward = cy > prev_y

            #print(f"[Debug] Vehicle ID={track_id}, prev=({prev_x},{prev_y}), curr=({cx},{cy}), was_outside={was_outside}, is_inside={is_inside}, downward={downward}")

            if was_outside and is_inside and downward:
                if track_id not in line_zone_manager.vehicle_crossed:
                    line_zone_manager.vehicle_crossed.add(track_id)
                    logger.log_vehicle(track_id, frame.copy(), class_name, speed=0, state=2, line_drawer=line_zone_manager)
                    #print(f"[Vehicle Entry] ID={track_id}, cx={cx}, cy={cy}")

            if is_inside:
                line_zone_manager.vehicle_activated.add(track_id)

            vehicle_memory_point[track_id] = (cx, cy)

        elif class_id == 0:
            prev_y = person_memory_y.get(track_id, cy - 5)
            print(f"[Debug] Person {track_id}, prev_y={prev_y}, cy={cy}, cx={cx}, line={line_zone_manager.PERSON_LINE_Y}")

            # 정확한 로그/캡처 조건: 위에서 아래로 진입 + x범위 포함
            if (145 < prev_y < 150) and cy >= 150 and (line_zone_manager.PERSON_X1 < cx < line_zone_manager.PERSON_X2):
                if track_id not in line_zone_manager.person_crossed:
                    line_zone_manager.person_crossed.add(track_id)
                    logger.log_person(track_id, frame.copy(), class_name, speed=0, state=1, line_drawer=line_zone_manager)
                    print(f"[Person Entry] ID={track_id}, cx={cx}, cy={cy}")

            # 감지는 cy 기준으로만 처리
            if (line_zone_manager.PERSON_X1 < cx < line_zone_manager.PERSON_X2) and (cy >= line_zone_manager.PERSON_LINE_Y):
                line_zone_manager.person_activated.add(track_id)

            person_memory_y[track_id] = cy

        #print(f"[Debug] Detected class_id={class_id}, label={class_name}")
    
    # Update activated sets to remove IDs no longer tracked
    active_person_ids = {int(obj[4]) for obj, cls_id in zip(tracked_objects, valid_classes) if int(cls_id) == 0}
    active_vehicle_ids = {int(obj[4]) for obj, cls_id in zip(tracked_objects, valid_classes) if int(cls_id) in VEHICLE_CLASS_IDS}

    line_zone_manager.person_activated.intersection_update(active_person_ids)
    line_zone_manager.vehicle_activated.intersection_update(active_vehicle_ids)

    if not line_zone_manager.person_activated and not line_zone_manager.vehicle_activated:
        if state in [1, 2, 3]:
            state = 4
            last_active_time = time.time()
        elif state == 4 and time.time() - last_active_time > 10:
            state = 0
    else:
        if line_zone_manager.person_activated and line_zone_manager.vehicle_activated:
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
