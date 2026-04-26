from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

FieldPoint = Tuple[float, float]
ImagePoint = Tuple[float, float]


@dataclass
class HomographyResult:
    matrix: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    approximate: bool


def collect_points_ui(frame: np.ndarray) -> List[ImagePoint]:
    """Simple click UI to collect calibration points from a frame.

    Controls:
    - Left click: add point
    - Backspace: remove last point
    - Enter: confirm
    - Esc/q: cancel
    """
    points: List[ImagePoint] = []
    window_name = "VelocityVision Calibration"

    canvas = frame.copy()

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((float(x), float(y)))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        display = canvas.copy()
        for idx, (x, y) in enumerate(points):
            cv2.circle(display, (int(x), int(y)), 5, (0, 255, 255), -1)
            cv2.putText(display, str(idx + 1), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(display, "Click 4-8 points | Enter=done | Backspace=undo | Esc=cancel", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):
            break
        if key in (8, 127) and points:
            points.pop()
        if key in (27, ord("q")):
            points = []
            break

    cv2.destroyWindow(window_name)
    return points


def compute_homography(img_pts: Sequence[ImagePoint], field_pts: Sequence[FieldPoint]) -> HomographyResult:
    if len(img_pts) < 4 or len(field_pts) < 4 or len(img_pts) != len(field_pts):
        return HomographyResult(matrix=None, mask=None, approximate=True)

    img = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)
    field = np.array(field_pts, dtype=np.float32).reshape(-1, 1, 2)
    h_matrix, mask = cv2.findHomography(img, field, method=cv2.RANSAC, ransacReprojThreshold=4.0)

    return HomographyResult(matrix=h_matrix, mask=mask, approximate=h_matrix is None)


def project_points(points: Sequence[ImagePoint], h_matrix: Optional[np.ndarray]) -> np.ndarray:
    if h_matrix is None or len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, h_matrix)
    return projected.reshape(-1, 2)
