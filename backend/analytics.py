from __future__ import annotations

import csv
import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover
    KMeans = None

BBox = Tuple[float, float, float, float]
LOGGER = logging.getLogger(__name__)


@dataclass
class PlayerSnapshot:
    speed_kmh: float
    team_label: Optional[str]


class PitchCalibrator:
    """Homography helper for image-to-pitch mapping."""

    def __init__(self, frame_size: Tuple[int, int]):
        self.frame_width, self.frame_height = frame_size
        self.pitch_length_m = 105.0
        self.pitch_width_m = 68.0
        self.h_matrix = self._build_default_homography()
        self.valid = self.h_matrix is not None

    def _build_default_homography(self) -> Optional[np.ndarray]:
        w, h = self.frame_width, self.frame_height
        if w < 20 or h < 20:
            return None

        src = np.float32(
            [
                [0.20 * w, 0.26 * h],
                [0.80 * w, 0.26 * h],
                [0.96 * w, 0.94 * h],
                [0.04 * w, 0.94 * h],
            ]
        )
        dst = np.float32(
            [
                [0.0, 0.0],
                [self.pitch_length_m, 0.0],
                [self.pitch_length_m, self.pitch_width_m],
                [0.0, self.pitch_width_m],
            ]
        )
        h_matrix = cv2.getPerspectiveTransform(src, dst)
        return h_matrix

    def image_to_world(self, point_xy: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        if not self.valid or self.h_matrix is None:
            return None

        point = np.array([[[point_xy[0], point_xy[1]]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, self.h_matrix)
        x_m, y_m = mapped[0, 0]
        if np.isnan(x_m) or np.isnan(y_m):
            return None

        x_m = float(np.clip(x_m, 0.0, self.pitch_length_m))
        y_m = float(np.clip(y_m, 0.0, self.pitch_width_m))
        return x_m, y_m


class MatchAnalytics:
    """Production-grade analytics: speed, teams, possession, passes, heatmaps."""

    def __init__(self, fps: float, frame_size: Tuple[int, int], trajectory_size: int = 35):
        self.fps = fps if fps > 0 else 25.0
        self.frame_width, self.frame_height = frame_size
        self.pixel_to_meter_fallback = 68.0 / max(1, self.frame_width)

        self.calibrator = PitchCalibrator(frame_size=frame_size)
        self.trajectories: Dict[int, Deque[Tuple[int, int]]] = defaultdict(lambda: deque(maxlen=trajectory_size))
        self.world_trajectories: Dict[int, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=trajectory_size))

        self.prev_centers_img: Dict[int, Tuple[float, float]] = {}
        self.prev_centers_world: Dict[int, Tuple[float, float]] = {}
        self.player_speed_history: Dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=24))
        self.player_distance_m: Dict[int, float] = defaultdict(float)
        self.activity_accumulator: Dict[int, float] = defaultdict(float)

        self.jersey_history: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=18))
        self.team_assignments: Dict[int, str] = {}

        self.current_owner: Optional[int] = None
        self.pass_count = 0
        self.pass_matrix: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        self.possession_frames: Dict[str, int] = defaultdict(int)
        self.possession_timeline: List[Tuple[int, str]] = []

        self.heatmap = np.zeros((68, 105), dtype=np.float32)
        self.total_frames = 0

    @staticmethod
    def _center(box: BBox) -> Tuple[float, float]:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    @staticmethod
    def _jersey_patch(frame: np.ndarray, box: BBox) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        jersey_y1 = y1 + int((y2 - y1) * 0.15)
        jersey_y2 = y1 + int((y2 - y1) * 0.55)
        patch = frame[jersey_y1:jersey_y2, x1:x2]
        if patch.size == 0:
            return None

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        return hsv.reshape(-1, 3).mean(axis=0)

    def _update_team_clusters(self) -> None:
        candidates: List[Tuple[int, np.ndarray]] = []
        for player_id, samples in self.jersey_history.items():
            if len(samples) < 4:
                continue
            mean_sample = np.mean(np.array(samples), axis=0)
            candidates.append((player_id, mean_sample))

        if len(candidates) < 2:
            return

        ids = np.array([item[0] for item in candidates], dtype=np.int32)
        features = np.array([item[1] for item in candidates], dtype=np.float32)

        if KMeans is not None:
            km = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = km.fit_predict(features)
        else:
            threshold = float(np.median(features[:, 0]))
            labels = (features[:, 0] > threshold).astype(np.int32)

        for idx, pid in enumerate(ids.tolist()):
            self.team_assignments[pid] = "Team A" if int(labels[idx]) == 0 else "Team B"

    def _team_of_player(self, player_id: int) -> Optional[str]:
        return self.team_assignments.get(player_id)

    def _world_distance_m(self, player_id: int, center_img: Tuple[float, float]) -> Tuple[float, Optional[Tuple[float, float]]]:
        current_world = self.calibrator.image_to_world(center_img)
        if current_world is not None and player_id in self.prev_centers_world:
            dist_m = math.dist(current_world, self.prev_centers_world[player_id])
            return dist_m, current_world

        if player_id in self.prev_centers_img:
            dist_px = math.dist(center_img, self.prev_centers_img[player_id])
            return dist_px * self.pixel_to_meter_fallback, current_world

        return 0.0, current_world

    def _assign_possession(
        self,
        tracks: Dict[int, BBox],
        ball_box: Optional[Tuple[float, float, float, float, float]],
    ) -> Optional[int]:
        if ball_box is None or not tracks:
            return self.current_owner

        ball_center = ((ball_box[0] + ball_box[2]) / 2.0, (ball_box[1] + ball_box[3]) / 2.0)

        nearest_player = None
        nearest_dist = float("inf")
        for track_id, box in tracks.items():
            d = math.dist(ball_center, self._center(box))
            if d < nearest_dist:
                nearest_dist = d
                nearest_player = track_id

        possession_radius_px = max(20.0, self.frame_width * 0.03)
        if nearest_player is None or nearest_dist > possession_radius_px:
            return self.current_owner

        if self.current_owner is not None and self.current_owner != nearest_player:
            self.pass_count += 1
            self.pass_matrix[self.current_owner][nearest_player] += 1

        self.current_owner = nearest_player
        return self.current_owner

    def _update_heatmap(self, world_points: Iterable[Tuple[float, float]]) -> None:
        for x_m, y_m in world_points:
            x_idx = int(np.clip(round(x_m), 0, self.heatmap.shape[1] - 1))
            y_idx = int(np.clip(round(y_m), 0, self.heatmap.shape[0] - 1))
            self.heatmap[y_idx, x_idx] += 1.0

    def update(
        self,
        frame: np.ndarray,
        tracks: Dict[int, BBox],
        ball_box: Optional[Tuple[float, float, float, float, float]],
        frame_idx: int = 0,
    ) -> Tuple[Dict[int, PlayerSnapshot], Optional[int]]:
        snapshots: Dict[int, PlayerSnapshot] = {}
        self.total_frames += 1
        world_points: List[Tuple[float, float]] = []

        for track_id, box in tracks.items():
            center = self._center(box)
            self.trajectories[track_id].append((int(center[0]), int(center[1])))

            dist_m, center_world = self._world_distance_m(track_id, center)
            speed_kmh = min(dist_m * self.fps * 3.6, 38.0)

            self.prev_centers_img[track_id] = center
            if center_world is not None:
                self.prev_centers_world[track_id] = center_world
                self.world_trajectories[track_id].append(center_world)
                world_points.append(center_world)

            self.player_distance_m[track_id] += dist_m
            self.activity_accumulator[track_id] += 0.65 * dist_m + 0.35 * speed_kmh
            self.player_speed_history[track_id].append(speed_kmh)

            jersey_mean = self._jersey_patch(frame, box)
            if jersey_mean is not None:
                self.jersey_history[track_id].append(jersey_mean)

        if frame_idx % 10 == 0:
            self._update_team_clusters()

        self._update_heatmap(world_points)

        owner = self._assign_possession(tracks, ball_box)
        owner_team = self._team_of_player(owner) if owner is not None else None
        if owner_team:
            self.possession_frames[owner_team] += 1
            self.possession_timeline.append((frame_idx, owner_team))

        for track_id in tracks:
            smooth_speed = float(np.mean(self.player_speed_history[track_id])) if self.player_speed_history[track_id] else 0.0
            snapshots[track_id] = PlayerSnapshot(
                speed_kmh=smooth_speed,
                team_label=self._team_of_player(track_id),
            )

        return snapshots, owner

    def _team_speed_stats(self) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for team in ["Team A", "Team B"]:
            team_players = [pid for pid, t in self.team_assignments.items() if t == team]
            team_speeds = [
                speed
                for pid in team_players
                for speed in self.player_speed_history.get(pid, [])
            ]
            stats[team] = {
                "avg_speed_kmh": round(float(np.mean(team_speeds)) if team_speeds else 0.0, 2),
                "max_speed_kmh": round(float(np.max(team_speeds)) if team_speeds else 0.0, 2),
                "players": len(team_players),
            }
        return stats

    def possession_percentages(self) -> Dict[str, float]:
        total_possession = max(1, sum(self.possession_frames.values()))
        return {
            "Team A": round(100.0 * self.possession_frames.get("Team A", 0) / total_possession, 2),
            "Team B": round(100.0 * self.possession_frames.get("Team B", 0) / total_possession, 2),
        }

    def summary(self) -> dict:
        all_speeds = [speed for speeds in self.player_speed_history.values() for speed in speeds]
        avg_speed = float(np.mean(all_speeds)) if all_speeds else 0.0
        max_speed = float(np.max(all_speeds)) if all_speeds else 0.0

        pass_edges = []
        for source, targets in self.pass_matrix.items():
            for target, count in targets.items():
                pass_edges.append({"from": int(source), "to": int(target), "count": int(count)})

        possession_pct = self.possession_percentages()

        player_rankings = []
        for pid, score in sorted(self.activity_accumulator.items(), key=lambda kv: kv[1], reverse=True):
            player_rankings.append(
                {
                    "player_id": int(pid),
                    "team": self.team_assignments.get(pid, "Unknown"),
                    "score": round(float(score), 2),
                    "distance_m": round(float(self.player_distance_m.get(pid, 0.0)), 2),
                    "max_speed_kmh": round(float(np.max(self.player_speed_history.get(pid, [0.0]))), 2),
                }
            )

        return {
            "avg_speed_kmh": round(avg_speed, 2),
            "max_speed_kmh": round(max_speed, 2),
            "pass_count": int(self.pass_count),
            "pass_edges": pass_edges,
            "players_tracked": len(self.player_speed_history),
            "team_speed_stats": self._team_speed_stats(),
            "possession_pct": possession_pct,
            "player_rankings": player_rankings[:15],
            "homography_enabled": bool(self.calibrator.valid),
        }

    def render_pass_network(self, output_path: str | Path) -> Optional[str]:
        if not self.pass_matrix:
            return None

        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except Exception:
            return None

        graph = nx.DiGraph()
        for src, targets in self.pass_matrix.items():
            for dst, count in targets.items():
                graph.add_edge(src, dst, weight=count)

        if graph.number_of_edges() == 0:
            return None

        node_colors = []
        for node in graph.nodes:
            team = self.team_assignments.get(node)
            if team == "Team A":
                node_colors.append("#22D3EE")
            elif team == "Team B":
                node_colors.append("#F97316")
            else:
                node_colors.append("#A78BFA")

        pos = nx.spring_layout(graph, seed=13)
        weights = [graph[u][v]["weight"] for u, v in graph.edges()]

        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(graph, pos, node_size=950, node_color=node_colors, edgecolors="#0b1020")
        nx.draw_networkx_labels(graph, pos, font_color="#0b1020")
        nx.draw_networkx_edges(graph, pos, arrowstyle="->", arrowsize=18, width=[1.5 + 1.2 * w for w in weights], edge_color="#CBD5E1")
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels={(u, v): graph[u][v]["weight"] for u, v in graph.edges()},
            font_color="#111827",
        )
        plt.title("Advanced Passing Network")
        plt.axis("off")
        output_path = str(output_path)
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return output_path

    def render_heatmap(self, output_path: str | Path) -> Optional[str]:
        if float(self.heatmap.sum()) <= 0:
            return None

        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None

        heat = cv2.GaussianBlur(self.heatmap, (0, 0), sigmaX=2.0, sigmaY=2.0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor("#0d1b2a")

        ax.imshow(heat, cmap="inferno", alpha=0.86, origin="lower", extent=[0, 105, 0, 68], aspect="auto")
        ax.plot([0, 105, 105, 0, 0], [0, 0, 68, 68, 0], color="white", linewidth=1.6)
        ax.axvline(x=52.5, color="white", linestyle="--", linewidth=1.0)
        circle = plt.Circle((52.5, 34), 9.15, color="white", fill=False, linewidth=1.0)
        ax.add_patch(circle)

        ax.set_title("Player Movement Heatmap", color="white", fontsize=14)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 68)
        ax.set_xticks([])
        ax.set_yticks([])

        output_path = str(output_path)
        plt.tight_layout()
        plt.savefig(output_path, dpi=160, facecolor="#0d1b2a")
        plt.close(fig)
        return output_path

    def render_possession_timeline(self, output_path: str | Path) -> Optional[str]:
        if not self.possession_timeline:
            return None

        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None

        frames = [item[0] for item in self.possession_timeline]
        teams = [item[1] for item in self.possession_timeline]
        encoded = [1 if t == "Team A" else 0 for t in teams]

        fig, ax = plt.subplots(figsize=(10, 2.8))
        ax.plot(frames, encoded, color="#22D3EE", linewidth=1.4)
        ax.fill_between(frames, encoded, color="#22D3EE", alpha=0.18)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Team B", "Team A"])
        ax.set_xlabel("Frame")
        ax.set_title("Possession Timeline")
        ax.grid(alpha=0.22)

        output_path = str(output_path)
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path

    def export_player_csv(self, output_path: str | Path) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["player_id", "team", "avg_speed_kmh", "max_speed_kmh", "distance_m", "activity_score"])
            for pid in sorted(self.player_speed_history.keys()):
                speeds = list(self.player_speed_history[pid])
                writer.writerow(
                    [
                        pid,
                        self.team_assignments.get(pid, "Unknown"),
                        round(float(np.mean(speeds)) if speeds else 0.0, 2),
                        round(float(np.max(speeds)) if speeds else 0.0, 2),
                        round(float(self.player_distance_m.get(pid, 0.0)), 2),
                        round(float(self.activity_accumulator.get(pid, 0.0)), 2),
                    ]
                )

        LOGGER.info("Exported player csv to %s", output_path)
        return str(output_path)
