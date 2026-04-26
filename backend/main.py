from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.analytics_pass import PossessionEngine
from backend.analytics_speed import compute_speed
from backend.ball_tracking import BallTracker
from backend.calibrate import compute_homography, project_points
from backend.detection import YOLODetector
from backend.filters import reject_jitter
from backend.metrics import MetricsEngine, render_heatmap, render_pass_network, render_possession_timeline
from backend.prediction import NextPassPredictor
from backend.team_id import TeamIdentifier
from backend.tracking import TrackerEngine
from backend.video_renderer import VideoRenderer

APP_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = APP_ROOT / "outputs"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
MODEL_DIR = APP_ROOT / "models"

for directory in (OUTPUT_DIR, UPLOAD_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger("velocityvision")


@dataclass
class JobState:
    status: str = "queued"
    progress: float = 0.0
    message: str = "Waiting to start"
    error: Optional[str] = None
    output_video: Optional[str] = None
    pass_graph: Optional[str] = None
    heatmap: Optional[str] = None
    possession_timeline: Optional[str] = None
    csv_export: Optional[str] = None
    metrics_json: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


jobs: Dict[str, JobState] = {}
job_lock = threading.Lock()

app = FastAPI(title="VelocityVision API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _set_job(job_id: str, **kwargs: Any) -> None:
    with job_lock:
        if job_id not in jobs:
            return
        for k, v in kwargs.items():
            setattr(jobs[job_id], k, v)


def _scale_box(box: Tuple[float, float, float, float, float], sx: float, sy: float):
    x1, y1, x2, y2, conf = box
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy, conf)


def _center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _load_homography(frame_w: int, frame_h: int) -> Tuple[Optional[np.ndarray], bool]:
    calib_file = os.getenv("VELOCITYVISION_CALIBRATION_JSON")
    if calib_file:
        path = Path(calib_file)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                img_pts = payload.get("img_pts", [])
                field_pts = payload.get("field_pts", [])
                result = compute_homography(img_pts, field_pts)
                if result.matrix is not None:
                    LOGGER.info("Loaded manual homography from %s", path)
                    return result.matrix, False
            except Exception:
                LOGGER.exception("Failed reading calibration file")

    # fallback approximation mapping frame corners -> pitch coordinates
    img_pts = [(0.0, 0.0), (float(frame_w - 1), 0.0), (float(frame_w - 1), float(frame_h - 1)), (0.0, float(frame_h - 1))]
    field_pts = [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0)]
    result = compute_homography(img_pts, field_pts)
    return result.matrix, True


def _project_to_metric(point_img: Tuple[float, float], h_matrix: Optional[np.ndarray]) -> Tuple[float, float]:
    if h_matrix is None:
        return (0.0, 0.0)
    proj = project_points([point_img], h_matrix)
    if len(proj) == 0:
        return (0.0, 0.0)
    x, y = float(np.clip(proj[0][0], 0.0, 105.0)), float(np.clip(proj[0][1], 0.0, 68.0))
    return x, y


def _process_video(job_id: str, video_path: Path, detect_stride: int = 1) -> None:
    cap = None
    renderer = None

    try:
        _set_job(job_id, status="processing", message="Loading models", progress=0.01)

        custom_model = MODEL_DIR / "yolov8n.pt"
        model_source = custom_model if custom_model.exists() else "yolov8n.pt"
        detector = YOLODetector(model_path=model_source, player_conf_threshold=0.3, ball_conf_threshold=0.12)
        tracker = TrackerEngine(iou_threshold=0.28, max_age=30, min_hits=3, high_thresh=0.42, low_thresh=0.1)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("Unable to open input video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        h_matrix, approx_metrics = _load_homography(width, height)

        team_id = TeamIdentifier(min_samples=4)
        ball_tracker = BallTracker(max_missing=8, max_jump_px=max(90.0, width * 0.12))
        poss_engine = PossessionEngine(owner_radius_m=2.3, stability_frames=6, pass_cooldown_frames=12, pass_min_ball_speed_kmh=7.0)
        predictor = NextPassPredictor()
        metrics = MetricsEngine(fps=fps)

        track_hist_metric: Dict[int, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=36))
        track_hist_img: Dict[int, Deque[Tuple[int, int]]] = defaultdict(lambda: deque(maxlen=36))
        last_speed_map: Dict[int, float] = defaultdict(float)

        heatmap = np.zeros((68, 105), dtype=np.float32)
        output_video = OUTPUT_DIR / f"{job_id}_annotated.mp4"
        renderer = VideoRenderer(output_video, fps=fps, frame_size=(width, height), output_size=(width, height))

        prev_ball_metric: Optional[Tuple[float, float]] = None
        frame_idx = 0
        processing_fps = 0.0
        last_tick = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            do_detect = frame_idx % max(1, detect_stride) == 0
            if do_detect:
                infer_frame = frame
                sx = sy = 1.0
                if width > 960:
                    infer_w = 960
                    infer_h = max(320, int(height * (infer_w / width)))
                    infer_frame = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
                    sx, sy = width / infer_w, height / infer_h

                detections = detector.detect(infer_frame)
                player_dets = [_scale_box(d, sx, sy) if (sx != 1.0 or sy != 1.0) else d for d in detections.players]
                ball_cands = [_scale_box(d, sx, sy) if (sx != 1.0 or sy != 1.0) else d for d in detections.ball_candidates]
                tracks = tracker.update(player_dets)
            else:
                ball_cands = []
                tracks = tracker.predict_only()

            track_bbox_map = {t.track_id: t.bbox for t in tracks if t.time_since_update <= 1}

            player_metric_positions: Dict[int, Tuple[float, float]] = {}
            player_img_centers: Dict[int, Tuple[int, int]] = {}

            for pid, bbox in track_bbox_map.items():
                c_img = _center(bbox)
                c_metric = _project_to_metric(c_img, h_matrix)

                if track_hist_metric[pid]:
                    c_metric = reject_jitter(track_hist_metric[pid][-1], c_metric, epsilon=0.03)

                track_hist_metric[pid].append(c_metric)
                track_hist_img[pid].append((int(c_img[0]), int(c_img[1])))
                player_metric_positions[pid] = c_metric
                player_img_centers[pid] = (int(c_img[0]), int(c_img[1]))

                # per-player speed from metric track history
                speeds, _outliers = compute_speed(list(track_hist_metric[pid]), fps=fps, max_speed_kmh=40.0)
                current_speed = float(np.mean(speeds[-5:])) if speeds else 0.0
                last_speed_map[pid] = current_speed

                step_dist = 0.0
                if len(track_hist_metric[pid]) >= 2:
                    p1 = np.array(track_hist_metric[pid][-2], dtype=np.float32)
                    p2 = np.array(track_hist_metric[pid][-1], dtype=np.float32)
                    step_dist = float(np.linalg.norm(p2 - p1))

                team_id.update_player(frame, pid, bbox)
                metrics.update_player(pid, current_speed, step_dist, team_id.team_of(pid))

                # heatmap
                hx = int(np.clip(round(c_metric[0]), 0, 104))
                hy = int(np.clip(round(c_metric[1]), 0, 67))
                heatmap[hy, hx] += 1.0

            if frame_idx % 10 == 0:
                team_id.update_clusters()

            ball_state = ball_tracker.update(ball_cands)
            ball_metric = _project_to_metric(ball_state.position, h_matrix) if ball_state.position is not None else None
            if ball_metric is None:
                ball_vel_mps = 0.0
                ball_vel_vec = (0.0, 0.0)
            else:
                if prev_ball_metric is None:
                    ball_vel_vec = (0.0, 0.0)
                else:
                    ball_vel_vec = ((ball_metric[0] - prev_ball_metric[0]) * fps, (ball_metric[1] - prev_ball_metric[1]) * fps)
                ball_vel_mps = float(np.hypot(ball_vel_vec[0], ball_vel_vec[1]))
                prev_ball_metric = ball_metric

            prev_owner = poss_engine.current_owner
            owner = poss_engine.assign_owner(ball_metric, player_metric_positions)
            pass_event = poss_engine.detect_pass(
                frame_idx=frame_idx,
                prev_owner=prev_owner,
                curr_owner=owner,
                ball_velocity_mps=ball_vel_mps,
                team_lookup=team_id.assignments,
            )
            poss_engine.update_possession_stats(frame_idx=frame_idx, owner=owner, team_lookup=team_id.assignments)

            pred = predictor.predict_next_receiver(
                ball_pos=ball_metric,
                ball_vel=ball_vel_vec,
                owner_id=owner,
                player_positions=player_metric_positions,
                player_teams=team_id.assignments,
            )

            pass_line = None
            if pass_event is not None:
                s = player_img_centers.get(pass_event.from_player)
                t = player_img_centers.get(pass_event.to_player)
                if s and t:
                    pass_line = (s, t)

            total_poss = max(1, poss_engine.possession_frames_by_team.get("Team A", 0) + poss_engine.possession_frames_by_team.get("Team B", 0))
            possession_pct = {
                "Team A": round(100.0 * poss_engine.possession_frames_by_team.get("Team A", 0) / total_poss, 2),
                "Team B": round(100.0 * poss_engine.possession_frames_by_team.get("Team B", 0) / total_poss, 2),
            }

            now = time.perf_counter()
            dt = max(1e-6, now - last_tick)
            last_tick = now
            inst_fps = 1.0 / dt
            processing_fps = inst_fps if processing_fps == 0 else 0.9 * processing_fps + 0.1 * inst_fps

            renderer.draw_frame(
                frame=frame,
                tracks=track_bbox_map,
                speed_map=dict(last_speed_map),
                team_map=dict(team_id.assignments),
                trajectories=track_hist_img,
                ball_pos=ball_state.position,
                owner_id=owner,
                pass_line=pass_line,
                predicted_receiver=pred.receiver_id,
                processing_fps=processing_fps,
                possession_pct=possession_pct,
            )

            frame_idx += 1
            if frame_idx % 5 == 0 or frame_idx == total_frames:
                progress = min(0.98, frame_idx / total_frames)
                _set_job(job_id, progress=progress, message=f"Processed {frame_idx}/{total_frames} frames")

        output_final = renderer.close()

        pass_edges = []
        for src, targets in poss_engine.pass_matrix.items():
            for dst, count in targets.items():
                pass_edges.append({"from": int(src), "to": int(dst), "count": int(count)})

        total_poss = max(1, poss_engine.possession_frames_by_team.get("Team A", 0) + poss_engine.possession_frames_by_team.get("Team B", 0))
        possession_pct = {
            "Team A": round(100.0 * poss_engine.possession_frames_by_team.get("Team A", 0) / total_poss, 2),
            "Team B": round(100.0 * poss_engine.possession_frames_by_team.get("Team B", 0) / total_poss, 2),
        }

        rankings = metrics.build_rankings(limit=20)
        summary = metrics.summary(
            pass_count=len(poss_engine.pass_events),
            possession_pct=possession_pct,
            pass_edges=pass_edges,
            player_rankings=rankings,
            approximate_metrics=approx_metrics,
        )
        validation = metrics.validate()
        summary["validation"] = {
            "fps_ok": validation.fps_ok,
            "speed_range_ok": validation.speed_range_ok,
            "notes": validation.notes,
        }

        pass_graph_path = OUTPUT_DIR / f"{job_id}_pass_network.png"
        heatmap_path = OUTPUT_DIR / f"{job_id}_heatmap.png"
        possession_path = OUTPUT_DIR / f"{job_id}_possession_timeline.png"
        csv_path = OUTPUT_DIR / f"{job_id}_player_rankings.csv"
        metrics_path = OUTPUT_DIR / f"{job_id}_metrics.json"

        rendered_graph = render_pass_network(pass_edges, dict(team_id.assignments), pass_graph_path)
        rendered_heatmap = render_heatmap(heatmap, heatmap_path)
        rendered_possession = render_possession_timeline(poss_engine.possession_timeline, possession_path)
        csv_export = metrics.export_csv(csv_path, rankings)
        metrics.export_json(metrics_path, summary)

        _set_job(
            job_id,
            status="completed",
            progress=1.0,
            message="Completed",
            output_video=str(output_final),
            pass_graph=rendered_graph,
            heatmap=rendered_heatmap,
            possession_timeline=rendered_possession,
            csv_export=csv_export,
            metrics_json=str(metrics_path),
            metrics=summary,
        )
        LOGGER.info("Job %s completed", job_id)
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Job %s failed", job_id)
        _set_job(job_id, status="failed", error=str(exc), message="Processing failed")
    finally:
        if cap is not None:
            cap.release()
        if renderer is not None:
            try:
                renderer.writer.release()
            except Exception:
                pass


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/process")
async def process_video(file: UploadFile = File(...), detect_stride: int = Query(default=1, ge=1, le=5)) -> Dict[str, str]:
    suffix = Path(file.filename or "input.mp4").suffix or ".mp4"
    if suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    job_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{job_id}{suffix}"

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    input_path.write_bytes(data)

    with job_lock:
        jobs[job_id] = JobState(status="queued", progress=0.0, message="Upload complete")

    worker = threading.Thread(target=_process_video, args=(job_id, input_path, detect_stride), daemon=True)
    worker.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> Dict[str, Any]:
    with job_lock:
        state = jobs.get(job_id)

    if not state:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job_id,
        "status": state.status,
        "progress": state.progress,
        "message": state.message,
        "error": state.error,
        "metrics": state.metrics,
        "has_video": bool(state.output_video and Path(state.output_video).exists()),
        "has_pass_graph": bool(state.pass_graph and Path(state.pass_graph).exists()),
        "has_heatmap": bool(state.heatmap and Path(state.heatmap).exists()),
        "has_possession_timeline": bool(state.possession_timeline and Path(state.possession_timeline).exists()),
        "has_csv": bool(state.csv_export and Path(state.csv_export).exists()),
    }


def _download_artifact(job_id: str, field: str, media_type: str, filename: str):
    with job_lock:
        state = jobs.get(job_id)

    if not state:
        raise HTTPException(status_code=404, detail="Job not found")

    target = getattr(state, field)
    if not target:
        raise HTTPException(status_code=404, detail="Artifact not available")

    path = Path(target)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing")

    return FileResponse(str(path), media_type=media_type, filename=filename)


@app.get("/api/jobs/{job_id}/download")
def download_result(job_id: str):
    return _download_artifact(job_id, "output_video", "video/mp4", f"velocityvision_{job_id}.mp4")


@app.get("/api/jobs/{job_id}/pass-network")
def download_pass_graph(job_id: str):
    return _download_artifact(job_id, "pass_graph", "image/png", f"velocityvision_pass_network_{job_id}.png")


@app.get("/api/jobs/{job_id}/heatmap")
def download_heatmap(job_id: str):
    return _download_artifact(job_id, "heatmap", "image/png", f"velocityvision_heatmap_{job_id}.png")


@app.get("/api/jobs/{job_id}/possession-timeline")
def download_possession_timeline(job_id: str):
    return _download_artifact(job_id, "possession_timeline", "image/png", f"velocityvision_possession_{job_id}.png")


@app.get("/api/jobs/{job_id}/player-csv")
def download_player_csv(job_id: str):
    return _download_artifact(job_id, "csv_export", "text/csv", f"velocityvision_players_{job_id}.csv")
