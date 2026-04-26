from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

BBox = Tuple[float, float, float, float]
Detection = Tuple[float, float, float, float, float]


@dataclass
class TrackResult:
    track_id: int
    bbox: BBox
    confidence: float
    age: int
    time_since_update: int


@dataclass
class _Track:
    track_id: int
    bbox: np.ndarray
    confidence: float
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))


def _center(box: Sequence[float]) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


class TrackerEngine:
    """ByteTrack-inspired tracker with high/low confidence association."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 24,
        min_hits: int = 3,
        high_thresh: float = 0.4,
        low_thresh: float = 0.1,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.next_id = 1
        self.tracks: List[_Track] = []

    def _predict_tracks(self) -> None:
        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1
            c = _center(track.bbox)
            w = track.bbox[2] - track.bbox[0]
            h = track.bbox[3] - track.bbox[1]
            pred_c = c + track.velocity
            track.bbox = np.array([pred_c[0] - w / 2.0, pred_c[1] - h / 2.0, pred_c[0] + w / 2.0, pred_c[1] + h / 2.0], dtype=np.float32)

    def _associate(self, track_indices: List[int], detections: List[Detection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not track_indices or not detections:
            return [], track_indices.copy(), list(range(len(detections)))

        mat = np.zeros((len(track_indices), len(detections)), dtype=np.float32)
        for i, t_idx in enumerate(track_indices):
            t = self.tracks[t_idx]
            for d_idx, det in enumerate(detections):
                mat[i, d_idx] = _iou(t.bbox, np.array(det[:4], dtype=np.float32))

        if linear_sum_assignment is not None:
            row, col = linear_sum_assignment(1.0 - mat)
            pair_candidates = list(zip(row.tolist(), col.tolist()))
        else:
            pair_candidates = []
            used_r: set[int] = set()
            used_c: set[int] = set()
            flat = [(mat[r, c], r, c) for r in range(mat.shape[0]) for c in range(mat.shape[1])]
            for score, r, c in sorted(flat, key=lambda x: x[0], reverse=True):
                if r in used_r or c in used_c:
                    continue
                pair_candidates.append((r, c))
                used_r.add(r)
                used_c.add(c)

        matches: List[Tuple[int, int]] = []
        used_tracks = set()
        used_dets = set()
        for r, c in pair_candidates:
            if mat[r, c] < self.iou_threshold:
                continue
            t_idx = track_indices[r]
            matches.append((t_idx, c))
            used_tracks.add(t_idx)
            used_dets.add(c)

        unmatched_tracks = [t for t in track_indices if t not in used_tracks]
        unmatched_dets = [d for d in range(len(detections)) if d not in used_dets]
        return matches, unmatched_tracks, unmatched_dets

    def _update_track(self, track: _Track, det: Detection) -> None:
        det_box = np.array(det[:4], dtype=np.float32)
        prev_c = _center(track.bbox)
        new_c = _center(det_box)
        track.velocity = 0.8 * track.velocity + 0.2 * (new_c - prev_c)
        track.bbox = det_box
        track.confidence = float(det[4])
        track.hits += 1
        track.time_since_update = 0

    def update(self, detections: List[Detection]) -> List[TrackResult]:
        self._predict_tracks()

        high = [d for d in detections if d[4] >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d[4] < self.high_thresh]

        active_indices = list(range(len(self.tracks)))
        matches_h, unmatched_tracks, unmatched_high = self._associate(active_indices, high)

        for t_idx, d_idx in matches_h:
            self._update_track(self.tracks[t_idx], high[d_idx])

        if unmatched_tracks and low:
            matches_l, unmatched_tracks, _ = self._associate(unmatched_tracks, low)
            for t_idx, d_idx in matches_l:
                self._update_track(self.tracks[t_idx], low[d_idx])

        # create tracks from unmatched high-confidence detections
        for d_idx in unmatched_high:
            det = high[d_idx]
            self.tracks.append(
                _Track(
                    track_id=self.next_id,
                    bbox=np.array(det[:4], dtype=np.float32),
                    confidence=float(det[4]),
                )
            )
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        outputs: List[TrackResult] = []
        for track in self.tracks:
            if track.hits >= self.min_hits or track.age <= self.min_hits:
                outputs.append(
                    TrackResult(
                        track_id=track.track_id,
                        bbox=tuple(track.bbox.tolist()),
                        confidence=track.confidence,
                        age=track.age,
                        time_since_update=track.time_since_update,
                    )
                )
        return outputs

    def predict_only(self) -> List[TrackResult]:
        return self.update([])
