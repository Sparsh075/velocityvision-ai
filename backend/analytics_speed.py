from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from backend.filters import savgol_smooth


def smooth_series(series: Sequence[float], method: str = "ema", alpha: float = 0.35) -> np.ndarray:
    arr = np.array(series, dtype=np.float32)
    if arr.size == 0:
        return arr

    if method == "savgol":
        return np.array(savgol_smooth(arr), dtype=np.float32)

    out = np.zeros_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = (1.0 - alpha) * out[i - 1] + alpha * arr[i]
    return out


def compute_speed(track_history_metric: Sequence[Tuple[float, float]], fps: float, max_speed_kmh: float = 40.0) -> Tuple[List[float], List[bool]]:
    if fps <= 0:
        fps = 25.0

    if len(track_history_metric) < 2:
        return [0.0] * len(track_history_metric), [False] * len(track_history_metric)

    speeds = [0.0]
    outliers = [False]
    dt = 1.0 / fps

    for i in range(1, len(track_history_metric)):
        p1 = np.array(track_history_metric[i - 1], dtype=np.float32)
        p2 = np.array(track_history_metric[i], dtype=np.float32)
        dist_m = float(np.linalg.norm(p2 - p1))
        speed_kmh = (dist_m / dt) * 3.6
        is_outlier = speed_kmh > max_speed_kmh
        if is_outlier:
            speed_kmh = max_speed_kmh
        speeds.append(speed_kmh)
        outliers.append(is_outlier)

    smooth = smooth_series(speeds, method="savgol")
    return [float(v) for v in smooth], outliers
