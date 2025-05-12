import sys
sys.path.insert(0, "/usr/lib/python3/dist-packages/")

import cv2
import time

def handle_vehicles(tracked_objects, classes, center_coords, frame, logger,
                    line_zone_manager, vehicle_logged_time, vehicle_memory_point,
                    CAPTURE_INTERVAL):
    """
    차량 객체 추적 및 감지 처리
    - 폴리곤 진입 + 외부 -> 내부 + 하향 이동일 때 캡처 및 로그 기록
    - 이전 프레임과 위치 비교로 상태 변경 판단
    - 캡처 감지된 차량이 있는 경우 True 반환
    """
    CLASS_NAMES = {
        2: "Car",
        3: "Motorcycle",
        5: "Bus",
        7: "Truck"
    }
    VEHICLE_CLASS_IDS = [2, 3, 5, 7]

    vehicle_detected = False
    current_ids = set()

    for obj, cls_id, (cx, cy) in zip(tracked_objects, classes, center_coords):
        if cls_id not in VEHICLE_CLASS_IDS:
            continue

        track_id = int(obj[4])
        current_ids.add(track_id)
        x1, y1, x2, y2 = map(int, obj[:4])
        class_name = CLASS_NAMES.get(cls_id, "Vehicle")

        prev_pt = vehicle_memory_point.get(track_id, (cx, cy - 1))
        is_inside = cv2.pointPolygonTest(line_zone_manager.POLYGON, (cx, cy), False) >= 0
        was_outside = cv2.pointPolygonTest(line_zone_manager.POLYGON, prev_pt, False) < 0
        downward = cy > prev_pt[1]

        if is_inside:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            line_zone_manager.vehicle_activated.add(track_id)

        now = time.time()
        if was_outside and is_inside and downward and (now - vehicle_logged_time.get(track_id, 0)) > CAPTURE_INTERVAL:
            logger.log_vehicle(track_id, frame.copy(), class_name, 0, 2, line_zone_manager)
            vehicle_logged_time[track_id] = now
            vehicle_detected = True

        vehicle_memory_point[track_id] = (cx, cy)

    # 현재 추적되지 않은 ID는 비활성화
    line_zone_manager.vehicle_activated.intersection_update(current_ids)

    return vehicle_detected
