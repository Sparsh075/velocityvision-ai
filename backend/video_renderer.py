from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import imageio_ffmpeg

BBox = Tuple[float, float, float, float]
TEAM_COLORS = {
    "Team A": (230, 180, 40),
    "Team B": (50, 120, 255),
    None: (0, 255, 180),
}


def _letterbox(frame, target_size: Tuple[int, int]):
    tw, th = target_size
    h, w = frame.shape[:2]
    scale = min(tw / max(1, w), th / max(1, h))
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(
        resized,
        top=(th - nh) // 2,
        bottom=th - nh - (th - nh) // 2,
        left=(tw - nw) // 2,
        right=tw - nw - (tw - nw) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=(10, 10, 10),
    )
    return canvas


class VideoRenderer:
    def __init__(self, output_path: str | Path, fps: float, frame_size: Tuple[int, int], output_size: Optional[Tuple[int, int]] = None):
        self.final_output_path = Path(output_path)
        self.temp_output_path = self.final_output_path.with_name(self.final_output_path.stem + "_temp.mp4")
        self.frame_width, self.frame_height = frame_size
        self.output_size = output_size or frame_size

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.temp_output_path), fourcc, fps, self.output_size)

    def draw_frame(
        self,
        frame,
        tracks: Dict[int, BBox],
        speed_map: Dict[int, float],
        team_map: Dict[int, str],
        trajectories: Dict[int, Iterable[Tuple[int, int]]],
        ball_pos: Optional[Tuple[float, float]],
        owner_id: Optional[int],
        pass_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
        predicted_receiver: Optional[int],
        processing_fps: float,
        possession_pct: Dict[str, float],
    ) -> None:
        canvas = frame.copy()

        for pid, box in tracks.items():
            x1, y1, x2, y2 = [int(v) for v in box]
            team = team_map.get(pid)
            color = TEAM_COLORS.get(team, TEAM_COLORS[None])
            if owner_id == pid:
                color = (30, 220, 255)

            speed = speed_map.get(pid, 0.0)
            label = f"ID {pid} | {team or 'Unknown'} | {speed:.1f} km/h"

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 2)

            pts = list(trajectories.get(pid, []))
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], color, 1)

            if predicted_receiver == pid:
                cv2.circle(canvas, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 15, (200, 255, 80), 2)

        if ball_pos is not None:
            bx, by = int(ball_pos[0]), int(ball_pos[1])
            cv2.circle(canvas, (bx, by), 8, (65, 105, 255), 2)
            cv2.putText(canvas, "BALL", (bx + 10, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (65, 105, 255), 2)

        if pass_line is not None:
            cv2.arrowedLine(canvas, pass_line[0], pass_line[1], (255, 255, 80), 2, tipLength=0.25)

        cv2.rectangle(canvas, (10, 10), (350, 120), (16, 16, 16), -1)
        cv2.rectangle(canvas, (10, 10), (350, 120), (120, 120, 120), 1)
        cv2.putText(canvas, f"Owner: {owner_id if owner_id is not None else '-'}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2)
        cv2.putText(canvas, f"Proc FPS: {processing_fps:.1f}", (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (240, 240, 240), 2)
        cv2.putText(canvas, f"Team A Poss: {possession_pct.get('Team A', 0):.1f}%", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEAM_COLORS["Team A"], 2)
        cv2.putText(canvas, f"Team B Poss: {possession_pct.get('Team B', 0):.1f}%", (185, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEAM_COLORS["Team B"], 2)

        out_frame = _letterbox(canvas, self.output_size) if self.output_size != (self.frame_width, self.frame_height) else canvas
        self.writer.write(out_frame)

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
