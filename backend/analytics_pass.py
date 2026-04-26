from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PassEvent:
    frame_idx: int
    from_player: int
    to_player: int
    team: Optional[str]


class PossessionEngine:
    def __init__(
        self,
        owner_radius_m: float = 2.2,
        stability_frames: int = 6,
        pass_cooldown_frames: int = 10,
        pass_min_ball_speed_kmh: float = 8.0,
    ):
        self.owner_radius_m = owner_radius_m
        self.stability_frames = stability_frames
        self.pass_cooldown_frames = pass_cooldown_frames
        self.pass_min_ball_speed_kmh = pass_min_ball_speed_kmh

        self.current_owner: Optional[int] = None
        self._candidate_owner: Optional[int] = None
        self._candidate_count = 0
        self.last_pass_frame = -10_000

        self.pass_matrix: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.pass_events: List[PassEvent] = []
        self.possession_frames_by_team: Dict[str, int] = defaultdict(int)
        self.possession_timeline: List[Tuple[int, str]] = []

    def assign_owner(
        self,
        ball_pos: Optional[Tuple[float, float]],
        player_positions_metric: Dict[int, Tuple[float, float]],
    ) -> Optional[int]:
        if ball_pos is None or not player_positions_metric:
            return self.current_owner

        nearest_id = None
        nearest_dist = float("inf")
        for pid, p in player_positions_metric.items():
            d = float(np.hypot(ball_pos[0] - p[0], ball_pos[1] - p[1]))
            if d < nearest_dist:
                nearest_dist = d
                nearest_id = pid

        if nearest_id is None or nearest_dist > self.owner_radius_m:
            return self.current_owner

        if self._candidate_owner == nearest_id:
            self._candidate_count += 1
        else:
            self._candidate_owner = nearest_id
            self._candidate_count = 1

        if self._candidate_count >= self.stability_frames:
            self.current_owner = self._candidate_owner

        return self.current_owner

    def detect_pass(
        self,
        frame_idx: int,
        prev_owner: Optional[int],
        curr_owner: Optional[int],
        ball_velocity_mps: float,
        team_lookup: Dict[int, str],
    ) -> Optional[PassEvent]:
        if prev_owner is None or curr_owner is None or prev_owner == curr_owner:
            return None

        if frame_idx - self.last_pass_frame < self.pass_cooldown_frames:
            return None

        if ball_velocity_mps * 3.6 < self.pass_min_ball_speed_kmh:
            return None

        self.last_pass_frame = frame_idx
        self.pass_matrix[prev_owner][curr_owner] += 1
        ev = PassEvent(
            frame_idx=frame_idx,
            from_player=prev_owner,
            to_player=curr_owner,
            team=team_lookup.get(prev_owner),
        )
        self.pass_events.append(ev)
        return ev

    def update_possession_stats(self, frame_idx: int, owner: Optional[int], team_lookup: Dict[int, str]) -> None:
        if owner is None:
            return
        team = team_lookup.get(owner)
        if team is None:
            return
        self.possession_frames_by_team[team] += 1
        self.possession_timeline.append((frame_idx, team))
