from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import imageio_ffmpeg

from backend.analytics import PlayerSnapshot

BBox = Tuple[float, float, float, float]
TEAM_COLORS = {
    "Team A": (230, 180, 40),
    "Team B": (50, 120, 255),
    None: (0, 255, 180),
}


class VideoRenderer:
    def __init__(self, output_path: str | Path, fps: float, frame_size: Tuple[int, int]):
        self.final_output_path = Path(output_path)
        self.temp_output_path = self.final_output_path.with_name(self.final_output_path.stem + "_temp.mp4")
        self.frame_width, self.frame_height = frame_size

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.temp_output_path),
            fourcc,
            fps,
            (self.frame_width, self.frame_height),
        )

    def draw_frame(
        self,
        frame,
        tracks: Dict[int, BBox],
        player_snapshots: Dict[int, PlayerSnapshot],
        trajectories: Dict[int, Iterable[Tuple[int, int]]],
        ball_box: Optional[Tuple[float, float, float, float, float]],
        owner: Optional[int],
        fps_overlay: float,
        possession_pct: Optional[Dict[str, float]] = None,
    ) -> None:
        for track_id, box in tracks.items():
            x1, y1, x2, y2 = [int(v) for v in box]
            snap = player_snapshots.get(track_id)
            team_label = snap.team_label if snap else None
            color = TEAM_COLORS.get(team_label, TEAM_COLORS[None])
            if owner == track_id:
                color = (30, 220, 255)

            speed_text = f"{snap.speed_kmh:.1f} km/h" if snap else "0.0 km/h"
            tag = f"ID {track_id}"
            if team_label:
                tag += f" | {team_label}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, tag, (x1, max(20, y1 - 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.51, color, 2)
            cv2.putText(frame, speed_text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 2)

            points = list(trajectories.get(track_id, []))
            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], color, 1)

        if ball_box is not None:
            bx1, by1, bx2, by2 = [int(v) for v in ball_box[:4]]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (50, 90, 255), 2)
            cv2.putText(frame, "BALL", (bx1, max(20, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 90, 255), 2)

        cv2.rectangle(frame, (10, 10), (330, 116), (18, 18, 18), -1)
        cv2.rectangle(frame, (10, 10), (330, 116), (120, 120, 120), 1)
        cv2.putText(frame, f"Possession: {owner if owner else '-'}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (240, 240, 240), 2)
        cv2.putText(frame, f"FPS: {fps_overlay:.1f}", (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (240, 240, 240), 2)

        if possession_pct:
            ta = possession_pct.get("Team A", 0.0)
            tb = possession_pct.get("Team B", 0.0)
            cv2.putText(frame, f"Team A: {ta:.1f}%", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEAM_COLORS["Team A"], 2)
            cv2.putText(frame, f"Team B: {tb:.1f}%", (170, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEAM_COLORS["Team B"], 2)

        self.writer.write(frame)

    def close(self) -> Path:
        self.writer.release()
        self.final_output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg_path,
            "-y",
            "-i",
            str(self.temp_output_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(self.final_output_path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if self.temp_output_path.exists():
            self.temp_output_path.unlink(missing_ok=True)

        return self.final_output_path
