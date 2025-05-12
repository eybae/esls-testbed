import os
import cv2
import time
import csv
import shutil

class Logger:
    def __init__(self, base_path="~/Dev/esls-testbed"):
        self.base_path = os.path.expanduser(base_path)
        os.makedirs(self.base_path, exist_ok=True)
        self.cleanup_old_data(hours=48)  # 48시간 지난 데이터 정리

    def cleanup_old_data(self, hours=48):
        cutoff = time.time() - hours * 3600

        # 로그 정리
        log_root = os.path.join(self.base_path, "log")
        self._remove_old_files(log_root, cutoff)

        # 캡처 정리
        capture_root = os.path.join(self.base_path, "capture")
        self._remove_old_files(capture_root, cutoff, recursive=True)

    def _remove_old_files(self, root_path, cutoff, recursive=False):
        for root, dirs, files in os.walk(root_path, topdown=False):
            for f in files:
                fpath = os.path.join(root, f)
                if os.path.getmtime(fpath) < cutoff:
                    os.remove(fpath)
            if recursive:
                for d in dirs:
                    dpath = os.path.join(root, d)
                    if not os.listdir(dpath):  # 비어 있으면 삭제
                        shutil.rmtree(dpath)

    def _log(self, track_id, frame, class_name, speed, obj_type, state=None, line_drawer=None):
        # draw 상태 라인 및 텍스트도 포함된 전체 결과 저장
        frame_to_save = frame.copy()
        if line_drawer and state is not None:
            line_drawer.draw(frame_to_save, state)

        now = time.localtime()
        year = time.strftime("%Y", now)
        month = time.strftime("%m", now)
        day = time.strftime("%d", now)
        hour = time.strftime("%H", now)
        minute = time.strftime("%M", now)
        second = time.strftime("%S", now)

        # log path: ~/Dev/esls-testbed/log/person(or vehicle)/YYYY/MM/DD.csv
        log_dir = os.path.join(self.base_path, 'log', obj_type, year, month)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{day}.csv")

        # capture path: ~/Dev/esls-testbed/capture/person(or vehicle)/YYYY/MM/DD/HH/mmss.jpg
        capture_dir = os.path.join(self.base_path, 'capture', obj_type, year, month, day, hour)
        os.makedirs(capture_dir, exist_ok=True)
        img_file = os.path.join(capture_dir, f"{minute}{second}.jpg")

        # log
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"{hour}:{minute}:{second}", track_id, class_name, speed])

        # capture
        if not os.path.exists(img_file):
            cv2.imwrite(img_file, frame_to_save)

    def log_person(self, track_id, frame, class_name, speed, state=None, line_drawer=None):
        self._log(track_id, frame, class_name, speed, obj_type="person", state=state, line_drawer=line_drawer)

    def log_vehicle(self, track_id, frame, class_name, speed, state=None, line_drawer=None):
        self._log(track_id, frame, class_name, speed, obj_type="vehicle", state=state, line_drawer=line_drawer)
