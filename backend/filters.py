from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter


def ema_smooth_point(prev: Tuple[float, float], current: Tuple[float, float], alpha: float = 0.35) -> Tuple[float, float]:
    return (
        (1.0 - alpha) * prev[0] + alpha * current[0],
        (1.0 - alpha) * prev[1] + alpha * current[1],
    )


def smooth_track_positions(
    history: Dict[int, Deque[Tuple[float, float]]],
    current_positions: Dict[int, Tuple[float, float]],
    alpha: float = 0.35,
) -> Dict[int, Tuple[float, float]]:
    out: Dict[int, Tuple[float, float]] = {}
    for pid, pos in current_positions.items():
        prev = history[pid][-1] if history[pid] else pos
        smooth = ema_smooth_point(prev, pos, alpha=alpha)
        history[pid].append(smooth)
        out[pid] = smooth
    return out


def reject_jitter(prev: Tuple[float, float], current: Tuple[float, float], epsilon: float = 0.08) -> Tuple[float, float]:
    if np.hypot(current[0] - prev[0], current[1] - prev[1]) < epsilon:
        return prev
    return current


def interpolate_short_gaps(series: Dict[int, Dict[int, Tuple[float, float]]], max_gap: int = 5) -> Dict[int, Dict[int, Tuple[float, float]]]:
    # series[player_id][frame] = (x, y)
    out = defaultdict(dict)
    for pid, samples in series.items():
        if not samples:
            continue
        keys = sorted(samples.keys())
        for i, frame_idx in enumerate(keys[:-1]):
            out[pid][frame_idx] = samples[frame_idx]
            nxt = keys[i + 1]
            gap = nxt - frame_idx
            if 1 < gap <= max_gap:
                p1 = np.array(samples[frame_idx], dtype=np.float32)
                p2 = np.array(samples[nxt], dtype=np.float32)
                for j in range(1, gap):
                    t = j / gap
                    p = (1.0 - t) * p1 + t * p2
                    out[pid][frame_idx + j] = (float(p[0]), float(p[1]))
        out[pid][keys[-1]] = samples[keys[-1]]
    return out


def savgol_smooth(values, window_length: int = 9, polyorder: int = 2):
    arr = np.array(list(values), dtype=np.float32)
    if arr.size < 5:
        return arr
    win = min(window_length, arr.size if arr.size % 2 == 1 else arr.size - 1)
    win = max(win, 5)
    if win <= polyorder:
        return arr
    return savgol_filter(arr, window_length=win, polyorder=polyorder)
