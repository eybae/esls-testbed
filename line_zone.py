import cv2
import numpy as np
from datetime import datetime
import time

class LineZoneManager:
    def __init__(self):
        # 라인 설정
        self.PERSON_LINE_Y = 110
        self.PERSON_X1 = 130
        self.PERSON_X2 = 480

        self.VEHICLE_LINE_Y = 55
        self.VEHICLE_X1 = 300
        self.VEHICLE_X2 = 640
        self.VEHICLE_STOP_X_THRESHOLD = 350

        # 차량 감지 영역 (폴리곤)
        self.POLYGON = np.array([
            [280, 480],
            [310, 55],
            [390, 55],
            [640, 220],
            [640, 480]
        ])

        self.person_crossed = set()
        self.vehicle_crossed = set()
        self.person_activated = set()
        self.vehicle_activated = set()
        
        self.last_seen_time = time.time()

    def draw(self, frame, state):
        # 사람 감지 라인 (초록)
        cv2.line(frame, (self.PERSON_X1, self.PERSON_LINE_Y), (self.PERSON_X2, self.PERSON_LINE_Y), (0, 255, 0), 2)
        #cv2.putText(frame, "Person Line", (self.PERSON_X1 + 5, self.PERSON_LINE_Y - 10),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 차량 감지 영역 (폴리곤 - 파랑)
        cv2.polylines(frame, [self.POLYGON], isClosed=True, color=(255, 0, 0), thickness=2)
        #cv2.putText(frame, "Vehicle Zone", (self.POLYGON[1][0] + 5, self.POLYGON[1][1] + 15),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 현재 상태 출력
        if state == 0:
            state_text = f"Lamp: OFF"
            cv2.putText(frame, state_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        else:
            state_text = f"Lamp: ON"
            cv2.putText(frame, state_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        state_text = f"State: {state}"
        cv2.putText(frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, now_str, (360, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        #cv2.rectangle(frame, (380, 85), (400, 70), (255, 0, 0), 2)
