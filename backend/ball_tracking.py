from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from backend.filters import ema_smooth_point

BBox5 = Tuple[float, float, float, float, float]


@dataclass
class BallState:
    position: Optional[Tuple[float, float]]
    confidence: float
    velocity: Tuple[float, float]


class BallTracker:
    """Temporal linking for ball detections with short-gap interpolation."""

    def __init__(self, max_missing: int = 8, max_jump_px: float = 140.0):
        self.max_missing = max_missing
        self.max_jump_px = max_jump_px
        self.history: Deque[Tuple[float, float]] = deque(maxlen=120)
        self.conf_history: Deque[float] = deque(maxlen=120)
        self.missing = 0
        self.last_position: Optional[Tuple[float, float]] = None
        self.last_velocity: Tuple[float, float] = (0.0, 0.0)

    @staticmethod
    def _center(box: BBox5) -> Tuple[float, float]:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def _predict(self) -> Optional[Tuple[float, float]]:
        if self.last_position is None:
            return None
        return (self.last_position[0] + self.last_velocity[0], self.last_position[1] + self.last_velocity[1])

    def _choose_candidate(self, candidates: List[BBox5]) -> Optional[Tuple[float, float, float]]:
        if not candidates:
            return None

        pred = self._predict()
        scored: List[Tuple[float, Tuple[float, float, float]]] = []
        for cand in candidates:
            cx, cy = self._center(cand)
            conf = float(cand[4])
            if pred is None:
                score = conf
            else:
                dist = float(np.hypot(cx - pred[0], cy - pred[1]))
                score = conf - 0.004 * dist
            scored.append((score, (cx, cy, conf)))

        best = max(scored, key=lambda x: x[0])[1]
        if pred is not None:
            jump = float(np.hypot(best[0] - pred[0], best[1] - pred[1]))
            if jump > self.max_jump_px:
                return None
        return best

    def update(self, candidates: List[BBox5]) -> BallState:
        chosen = self._choose_candidate(candidates)

        if chosen is None:
            self.missing += 1
            if self.last_position is not None and self.missing <= self.max_missing:
                # short gap interpolation by constant velocity
                px = self.last_position[0] + self.last_velocity[0]
                py = self.last_position[1] + self.last_velocity[1]
                self.last_position = (px, py)
                self.history.append(self.last_position)
                self.conf_history.append(0.0)
                return BallState(position=self.last_position, confidence=0.0, velocity=self.last_velocity)

            self.last_position = None
            self.last_velocity = (0.0, 0.0)
            return BallState(position=None, confidence=0.0, velocity=(0.0, 0.0))

        x, y, conf = chosen
        if self.last_position is None:
            smooth = (x, y)
            vel = (0.0, 0.0)
        else:
            smooth = ema_smooth_point(self.last_position, (x, y), alpha=0.45)
            vel = (smooth[0] - self.last_position[0], smooth[1] - self.last_position[1])

        self.last_position = smooth
        self.last_velocity = vel
        self.missing = 0
        self.history.append(smooth)
        self.conf_history.append(conf)

        return BallState(position=smooth, confidence=conf, velocity=vel)
