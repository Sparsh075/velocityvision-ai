from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ValidationReport:
    fps_ok: bool
    speed_range_ok: bool
    notes: List[str]


class MetricsEngine:
    def __init__(self, fps: float):
        self.fps = fps if fps > 0 else 25.0
        self.player_speeds: Dict[int, List[float]] = defaultdict(list)
        self.player_dist_m: Dict[int, float] = defaultdict(float)
        self.player_activity: Dict[int, float] = defaultdict(float)
        self.team_map: Dict[int, str] = {}

    def update_player(self, player_id: int, speed_kmh: float, distance_m: float, team: Optional[str]) -> None:
        self.player_speeds[player_id].append(float(speed_kmh))
        self.player_dist_m[player_id] += float(distance_m)
        self.player_activity[player_id] += 0.6 * float(distance_m) + 0.4 * float(speed_kmh)
        if team:
            self.team_map[player_id] = team

    def build_rankings(self, limit: int = 20) -> List[dict]:
        rows = []
        for pid, score in sorted(self.player_activity.items(), key=lambda kv: kv[1], reverse=True):
            speeds = self.player_speeds.get(pid, [])
            rows.append(
                {
                    "player_id": int(pid),
                    "team": self.team_map.get(pid, "Unknown"),
                    "score": round(float(score), 2),
                    "distance_m": round(float(self.player_dist_m.get(pid, 0.0)), 2),
                    "avg_speed_kmh": round(float(np.mean(speeds)) if speeds else 0.0, 2),
                    "max_speed_kmh": round(float(np.max(speeds)) if speeds else 0.0, 2),
                }
            )
        return rows[:limit]

    def summary(self, pass_count: int, possession_pct: Dict[str, float], pass_edges: List[dict], player_rankings: List[dict], approximate_metrics: bool) -> dict:
        all_speeds = [s for series in self.player_speeds.values() for s in series]
        avg_speed = float(np.mean(all_speeds)) if all_speeds else 0.0
        max_speed = float(np.max(all_speeds)) if all_speeds else 0.0

        team_stats = {}
        for team in ["Team A", "Team B"]:
            team_players = [pid for pid, t in self.team_map.items() if t == team]
            team_speeds = [s for pid in team_players for s in self.player_speeds.get(pid, [])]
            team_stats[team] = {
                "avg_speed_kmh": round(float(np.mean(team_speeds)) if team_speeds else 0.0, 2),
                "max_speed_kmh": round(float(np.max(team_speeds)) if team_speeds else 0.0, 2),
                "players": len(team_players),
            }

        return {
            "avg_speed_kmh": round(avg_speed, 2),
            "max_speed_kmh": round(max_speed, 2),
            "pass_count": int(pass_count),
            "pass_edges": pass_edges,
            "team_speed_stats": team_stats,
            "possession_pct": possession_pct,
            "player_rankings": player_rankings,
            "players_tracked": len(self.player_speeds),
            "metrics_approximate": bool(approximate_metrics),
        }

    def validate(self) -> ValidationReport:
        notes: List[str] = []

        fps_ok = 8.0 <= self.fps <= 120.0
        if not fps_ok:
            notes.append(f"FPS out of expected range: {self.fps:.2f}")

        all_speeds = [s for series in self.player_speeds.values() for s in series]
        if not all_speeds:
            return ValidationReport(fps_ok=fps_ok, speed_range_ok=True, notes=notes)

        pct_high = float(np.mean(np.array(all_speeds) > 40.0))
        speed_range_ok = pct_high < 0.03
        if not speed_range_ok:
            notes.append("More than 3% speeds are above 40 km/h; check calibration")

        return ValidationReport(fps_ok=fps_ok, speed_range_ok=speed_range_ok, notes=notes)

    @staticmethod
    def export_json(path: str | Path, payload: dict) -> str:
        path = Path(path)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)

    @staticmethod
    def export_csv(path: str | Path, rows: List[dict]) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not rows:
            with path.open("w", encoding="utf-8") as f:
                f.write("player_id,team,score,distance_m,avg_speed_kmh,max_speed_kmh\n")
            return str(path)

        headers = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        return str(path)


def render_pass_network(pass_edges: List[dict], team_map: Dict[int, str], output_path: str | Path) -> Optional[str]:
    if not pass_edges:
        return None
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except Exception:
        return None

    graph = nx.DiGraph()
    for edge in pass_edges:
        graph.add_edge(edge["from"], edge["to"], weight=edge["count"])

    if graph.number_of_edges() == 0:
        return None

    node_colors = []
    for node in graph.nodes:
        team = team_map.get(node)
        if team == "Team A":
            node_colors.append("#22D3EE")
        elif team == "Team B":
            node_colors.append("#FB923C")
        else:
            node_colors.append("#A78BFA")

    pos = nx.spring_layout(graph, seed=42)
    weights = [graph[u][v]["weight"] for u, v in graph.edges()]

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(graph, pos, node_size=900, node_color=node_colors, edgecolors="#0f172a")
    nx.draw_networkx_labels(graph, pos)
    nx.draw_networkx_edges(graph, pos, arrowstyle="->", arrowsize=18, width=[1 + 1.2 * w for w in weights], edge_color="#cbd5e1")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels={(u, v): graph[u][v]["weight"] for u, v in graph.edges()})
    plt.title("Passing Network")
    plt.axis("off")
    output_path = str(output_path)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def render_heatmap(heatmap: np.ndarray, output_path: str | Path) -> Optional[str]:
    if heatmap.size == 0 or float(heatmap.sum()) <= 0:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    smooth = heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(smooth, cmap="inferno", origin="lower", extent=[0, 105, 0, 68], aspect="auto", alpha=0.9)
    plt.plot([0, 105, 105, 0, 0], [0, 0, 68, 68, 0], color="white", linewidth=1.5)
    plt.axvline(52.5, color="white", linestyle="--", linewidth=1)
    plt.title("Movement Heatmap")
    plt.xticks([])
    plt.yticks([])
    output_path = str(output_path)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, facecolor="#0f172a")
    plt.close()
    return output_path


def render_possession_timeline(timeline: List[Tuple[int, str]], output_path: str | Path) -> Optional[str]:
    if not timeline:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    frames = [t[0] for t in timeline]
    encoded = [1 if t[1] == "Team A" else 0 for t in timeline]

    plt.figure(figsize=(10, 2.8))
    plt.plot(frames, encoded, color="#22d3ee", linewidth=1.4)
    plt.fill_between(frames, encoded, color="#22d3ee", alpha=0.22)
    plt.yticks([0, 1], ["Team B", "Team A"])
    plt.xlabel("Frame")
    plt.title("Possession Timeline")
    plt.grid(alpha=0.2)
    output_path = str(output_path)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path
