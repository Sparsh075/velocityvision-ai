from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
except Exception:  # pragma: no cover
    RandomForestClassifier = None


@dataclass
class PredictionResult:
    receiver_id: Optional[int]
    probability: float
    ranked_candidates: List[Tuple[int, float]]


class NextPassPredictor:
    """Lightweight next-pass predictor with interpretable geometric features."""

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=120, random_state=42) if RandomForestClassifier else None
        self.is_trained = False

    @staticmethod
    def _feature(ball_pos, ball_vel, owner_pos, cand_pos):
        bx, by = ball_pos
        vx, vy = ball_vel
        ox, oy = owner_pos
        cx, cy = cand_pos

        to_cand = np.array([cx - ox, cy - oy], dtype=np.float32)
        dist = float(np.linalg.norm(to_cand))
        angle = float(np.arctan2(to_cand[1], to_cand[0]))

        vel = np.array([vx, vy], dtype=np.float32)
        vel_norm = float(np.linalg.norm(vel))
        dir_align = 0.0
        if vel_norm > 1e-4 and dist > 1e-4:
            dir_align = float(np.dot(vel, to_cand) / (vel_norm * dist))

        return [bx, by, vx, vy, ox, oy, cx, cy, dist, angle, dir_align]

    def train(self, X: List[List[float]], y: List[int]) -> None:
        if self.model is None or not X or not y:
            return
        self.model.fit(X, y)
        self.is_trained = True

    def predict_next_receiver(
        self,
        ball_pos: Optional[Tuple[float, float]],
        ball_vel: Tuple[float, float],
        owner_id: Optional[int],
        player_positions: Dict[int, Tuple[float, float]],
        player_teams: Dict[int, str],
    ) -> PredictionResult:
        if ball_pos is None or owner_id is None or owner_id not in player_positions:
            return PredictionResult(receiver_id=None, probability=0.0, ranked_candidates=[])

        owner_team = player_teams.get(owner_id)
        owner_pos = player_positions[owner_id]

        candidates = []
        for pid, pos in player_positions.items():
            if pid == owner_id:
                continue
            if owner_team and player_teams.get(pid) != owner_team:
                continue
            candidates.append((pid, pos))

        if not candidates:
            return PredictionResult(receiver_id=None, probability=0.0, ranked_candidates=[])

        feats = [self._feature(ball_pos, ball_vel, owner_pos, pos) for _, pos in candidates]

        if self.is_trained and self.model is not None:
            probs = self.model.predict_proba(feats)
            cls_index = {c: i for i, c in enumerate(self.model.classes_)}
            ranked = []
            for (pid, _), p in zip(candidates, probs):
                idx = cls_index.get(pid)
                pr = float(p[idx]) if idx is not None else 0.0
                ranked.append((pid, pr))
            ranked.sort(key=lambda x: x[1], reverse=True)
            return PredictionResult(receiver_id=ranked[0][0], probability=ranked[0][1], ranked_candidates=ranked)

        # heuristic fallback when offline model not trained
        heur = []
        for pid, pos in candidates:
            f = self._feature(ball_pos, ball_vel, owner_pos, pos)
            dist = f[8]
            align = f[10]
            score = (1.0 / (1.0 + dist)) * 0.6 + ((align + 1.0) / 2.0) * 0.4
            heur.append((pid, float(score)))

        heur.sort(key=lambda x: x[1], reverse=True)
        return PredictionResult(receiver_id=heur[0][0], probability=heur[0][1], ranked_candidates=heur)
