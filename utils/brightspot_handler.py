import sys
sys.path.insert(0, "/usr/lib/python3/dist-packages/")

import time
import cv2

def handle_brightspots(night, model_bright, frame, line_zone_manager,
                        bright_logged_time, CAPTURE_INTERVAL, logger, bright_recent_y):
    """
    야간 밝은 점 감지 처리 함수
    - 방향성 있는 이동이 감지되고 차량 폴리곤 내에 위치한 경우에만 캡처
    - 캡처 간격 제한 적용
    """
    bright_spot_detected = False

    if not night:
        return False

    results_bright = model_bright.predict(frame, conf=0.25, task="detect")
    if results_bright and results_bright[0].boxes is not None:
        for box in results_bright[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0])
            if cls_id != 0:
                continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            coord_key = "bright_0"

            bright_recent_y.setdefault(coord_key, []).append(cy)
            if len(bright_recent_y[coord_key]) > 3:
                bright_recent_y[coord_key].pop(0)

            avg_prev_y = sum(bright_recent_y[coord_key][:-1]) / max(1, len(bright_recent_y[coord_key]) - 1)
            downward = cy - avg_prev_y > 3
            is_inside = cv2.pointPolygonTest(line_zone_manager.POLYGON, (cx, cy), False) >= 0

            if is_inside:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "bright_spot", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                bright_spot_detected = True

            now = time.time()
            last_time = bright_logged_time.get(coord_key, 0)
            if is_inside and downward and (now - last_time > CAPTURE_INTERVAL):
                logger.log_vehicle(-99, frame.copy(), "bright_spot", 0, 2, line_zone_manager)
                bright_logged_time[coord_key] = now

    return bright_spot_detected