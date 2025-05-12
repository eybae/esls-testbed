# tracker.py

import numpy as np

class Sort:
    def __init__(self):
        self.track_id = 0
        self.tracks = {}

    def update(self, detections):
        results = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            self.track_id += 1
            self.tracks[self.track_id] = (x1, y1, x2, y2)
            results.append([x1, y1, x2, y2, self.track_id])
        return results
