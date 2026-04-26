from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0
SPORTS_BALL_CLASS_ID = 32

Detection = Tuple[float, float, float, float, float]


@dataclass
class FrameDetections:
    players: List[Detection]
    ball: Optional[Detection]
    ball_candidates: List[Detection]


class YOLODetector:
    """YOLOv8 detector for players + ball with separate confidence thresholds."""

    def __init__(
        self,
        model_path: str | Path = "models/yolov8n.pt",
        player_conf_threshold: float = 0.3,
        ball_conf_threshold: float = 0.15,
    ):
        self.model_path = Path(model_path)
        self.player_conf_threshold = player_conf_threshold
        self.ball_conf_threshold = ball_conf_threshold
        self.model = YOLO(str(self.model_path))

    def detect(self, frame: np.ndarray) -> FrameDetections:
        result = self.model.predict(frame, conf=min(self.player_conf_threshold, self.ball_conf_threshold), verbose=False, device="cpu")[0]

        players: List[Detection] = []
        balls: List[Detection] = []

        if result.boxes is None:
            return FrameDetections(players=players, ball=None, ball_candidates=[])

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = box.tolist()
            det = (x1, y1, x2, y2, float(conf))
            if cls_id == PERSON_CLASS_ID and conf >= self.player_conf_threshold:
                players.append(det)
            elif cls_id == SPORTS_BALL_CLASS_ID and conf >= self.ball_conf_threshold:
                balls.append(det)

        best_ball = max(balls, key=lambda x: x[4]) if balls else None
        return FrameDetections(players=players, ball=best_ball, ball_candidates=balls)
