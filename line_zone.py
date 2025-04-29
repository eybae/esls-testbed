# ✅ 수정 완료된 line_zone.py - 방향성까지 반영 최종 버전
import sys
sys.path.insert(0, "/usr/lib/python3/dist-packages/") 

import cv2
from datetime import datetime

class LineZoneManager:
    def __init__(self):
        self.person_activated = set()
        self.vehicle_activated = set()
        self.person_crossed = set()
        self.vehicle_crossed = set()

        self.person_memory_y = dict()
        self.vehicle_memory_y = dict()

        self.PERSON_LINE_Y = 110
        self.PERSON_X1 = 130
        self.PERSON_X2 = 450

        self.VEHICLE_LINE_Y = 70
        self.VEHICLE_X1 = 300
        self.VEHICLE_X2 = 640
        self.VEHICLE_STOP_X_THRESHOLD = 350

    def check_person_cross(self, track_id, cx, cy):
        if track_id in self.person_crossed:
            return False

        if cy >= self.PERSON_LINE_Y:
            self.person_crossed.add(track_id)
            self.person_activated.add(track_id)
            return True
        return False

    def check_vehicle_cross(self, track_id, cx, cy):
        prev_y = self.vehicle_memory_y.get(track_id, None)

        if prev_y is None:
            self.vehicle_memory_y[track_id] = cy
            return False

        if track_id in self.vehicle_crossed:
            return False

        # 방향성 + crossing 기준
        if prev_y <= self.VEHICLE_LINE_Y and cy > self.VEHICLE_LINE_Y and prev_y < cy:
            self.vehicle_crossed.add(track_id)
            self.vehicle_activated.add(track_id)
            print(f"[Vehicle Crossing ✅ 아래 방향] Track ID: {track_id}")
            return True

        self.vehicle_memory_y[track_id] = cy
        return False

    def draw(self, frame, state):
        cv2.line(frame, (self.PERSON_X1, self.PERSON_LINE_Y),
                 (self.PERSON_X2, self.PERSON_LINE_Y), (0, 255, 0), 2)
        cv2.line(frame, (self.VEHICLE_X1, self.VEHICLE_LINE_Y),
                 (self.VEHICLE_X2, self.VEHICLE_LINE_Y), (255, 0, 0), 2)
        cv2.putText(frame, f"State: {state}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, now_str, (360, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)