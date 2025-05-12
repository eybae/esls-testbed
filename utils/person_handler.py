import sys
sys.path.insert(0, "/usr/lib/python3/dist-packages/")

import cv2
import time

def handle_persons(tracked_objects, classes, center_coords, frame, logger,
                   line_zone_manager, person_logged_time, person_inside_memory,
                   CAPTURE_INTERVAL, person_recent_y):
    """
    사람 객체 감지 후:
    - 감지 영역 내 진입 판단 및 캡처
    - 위에서 아래로 이동한 경우에만 캡처
    - 활성 ID 반환
    """
    CLASS_NAMES = {0: "Person"}
    detected = False
    current_ids = set()

    for obj, cls_id, (cx, cy) in zip(tracked_objects, classes, center_coords):
        if cls_id != 0:
            continue

        track_id = int(obj[4])
        current_ids.add(track_id)
        x1, y1, x2, y2 = map(int, obj[:4])
        class_name = CLASS_NAMES.get(cls_id, "Unknown")

        coord_key = f"person_{track_id}"
        is_inside = cv2.pointPolygonTest(line_zone_manager.PERSON_POLYGON, (cx, cy), False) >= 0
        was_inside = person_inside_memory.get(track_id, False)
        person_inside_memory[track_id] = is_inside

        # 방향성 판단용 Y좌표 기록
        person_recent_y.setdefault(track_id, []).append(cy)
        if len(person_recent_y[track_id]) > 3:
            person_recent_y[track_id].pop(0)

        avg_prev_y = sum(person_recent_y[track_id][:-1]) / max(1, len(person_recent_y[track_id]) - 1)
        downward = cy - avg_prev_y > 3

        now = time.time()
        in_capture_zone = cy < line_zone_manager.PERSON_LINE_Y + 5
        interval_ok = (now - person_logged_time.get(coord_key, 0)) > CAPTURE_INTERVAL

        if is_inside:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            line_zone_manager.person_activated.add(track_id)

        if not was_inside and is_inside and in_capture_zone and downward and interval_ok:
            logger.log_person(track_id, frame.copy(), class_name, 0, 1, line_zone_manager)
            person_logged_time[coord_key] = now
            detected = True

    line_zone_manager.person_activated.intersection_update(current_ids)
    return detected
