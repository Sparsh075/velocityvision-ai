from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover
    KMeans = None

BBox = Tuple[float, float, float, float]


class TeamIdentifier:
    def __init__(self, min_samples: int = 4):
        self.min_samples = min_samples
        self.samples: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=25))
        self.assignments: Dict[int, str] = {}

    def _extract_feature(self, frame: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        y_top = y1 + int((y2 - y1) * 0.15)
        y_bottom = y1 + int((y2 - y1) * 0.55)
        roi = frame[y_top:y_bottom, x1:x2]
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        return hsv.reshape(-1, 3).mean(axis=0)

    def update_player(self, frame: np.ndarray, track_id: int, bbox: BBox) -> None:
        feat = self._extract_feature(frame, bbox)
        if feat is not None:
            self.samples[track_id].append(feat)

    def update_clusters(self) -> None:
        feats = []
        ids = []
        for pid, vals in self.samples.items():
            if len(vals) < self.min_samples:
                continue
            ids.append(pid)
            feats.append(np.mean(np.array(vals), axis=0))

        if len(feats) < 2:
            return

        X = np.array(feats, dtype=np.float32)
        if KMeans is not None:
            km = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = km.fit_predict(X)
        else:
            med = float(np.median(X[:, 0]))
            labels = (X[:, 0] > med).astype(int)

        for pid, label in zip(ids, labels.tolist()):
            self.assignments[pid] = "Team A" if label == 0 else "Team B"

    def team_of(self, player_id: int) -> Optional[str]:
        return self.assignments.get(player_id)
