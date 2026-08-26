"""Lightweight overlay drawing for hand observations (debug / validation only)."""
from __future__ import annotations

import cv2
import numpy as np

from .wilor_estimator import HAND_BONES, HandObservation, WRIST

_LEFT = (0, 165, 255)   # orange (BGR)
_RIGHT = (0, 255, 0)    # green


def _project(points_cam: np.ndarray, focal: float, center: tuple[float, float]) -> np.ndarray:
    """Pinhole-project camera-frame 3D points to pixels. points_cam: (N,3)."""
    cx, cy = center
    z = np.clip(points_cam[:, 2], 1e-4, None)
    u = focal * points_cam[:, 0] / z + cx
    v = focal * points_cam[:, 1] / z + cy
    return np.stack([u, v], axis=1)


def draw_hands(image_bgr: np.ndarray, hands: list[HandObservation]) -> np.ndarray:
    """Return a copy of ``image_bgr`` with skeletons, bboxes, wrist axes and info text."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    center = (w / 2.0, h / 2.0)

    for hand in hands:
        color = _RIGHT if hand.is_right else _LEFT
        x1, y1, x2, y2 = hand.bbox.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        kp = hand.keypoints_2d.astype(int)
        for a, b in HAND_BONES:
            cv2.line(out, tuple(kp[a]), tuple(kp[b]), color, 2)
        for (x, y) in kp:
            cv2.circle(out, (int(x), int(y)), 3, (255, 255, 255), -1)

        # wrist 6DOF: project a small coordinate frame (x=red, y=green, z=blue).
        axis_len = 0.05  # meters
        origin = hand.wrist_pos_cam
        axes_cam = origin[None, :] + axis_len * hand.wrist_rotmat.T  # columns are basis vectors
        pts = _project(np.vstack([origin[None, :], axes_cam]), hand.focal_length, center)
        o = tuple(pts[0].astype(int))
        for i, axcol in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            cv2.line(out, o, tuple(pts[i + 1].astype(int)), axcol, 2)

        label = f"{'R' if hand.is_right else 'L'} pinch={hand.pinch_dist * 1000:.0f}mm"
        cv2.putText(out, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        wp = hand.wrist_pos_cam
        cv2.putText(out, f"wrist(m) {wp[0]:+.2f},{wp[1]:+.2f},{wp[2]:+.2f}",
                    (x1, min(h - 4, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def highlight_hand(image_bgr: np.ndarray, hand: HandObservation) -> None:
    """Draw a thick yellow box marking the selected controlling hand (in place)."""
    x1, y1, x2, y2 = hand.bbox.astype(int)
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 255), 3)


def draw_hud(image_bgr: np.ndarray, lines: list[str], org: tuple[int, int] = (10, 24)) -> None:
    """Draw stacked HUD text lines with a dark backdrop (in place)."""
    x, y0 = org
    for i, line in enumerate(lines):
        y = y0 + i * 24
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image_bgr, (x - 4, y - th - 4), (x + tw + 4, y + 6), (0, 0, 0), -1)
        cv2.putText(image_bgr, line, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_dof_bars(image_bgr: np.ndarray, values: np.ndarray, names: tuple[str, ...],
                  org: tuple[int, int] = (10, 300), width: int = 150,
                  raw: np.ndarray | None = None) -> None:
    """Draw one horizontal bar per DOF (values in [0,1], 1 = fully closed) in place.

    ``raw`` optionally appends the uncalibrated angle in radians after each ratio.
    """
    x, y0 = org
    for i, (name, v) in enumerate(zip(names, np.asarray(values, dtype=float))):
        y = y0 + i * 20
        cv2.putText(image_bgr, f"{name:>10s}", (x, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        bx = x + 84
        cv2.rectangle(image_bgr, (bx, y), (bx + width, y + 13), (60, 60, 60), -1)
        fill = int(width * float(np.clip(v, 0.0, 1.0)))
        cv2.rectangle(image_bgr, (bx, y), (bx + fill, y + 13), (0, 200, 255), -1)
        label = f"{v:.2f}" if raw is None else f"{v:.2f} {float(raw[i]):+.2f}r"
        cv2.putText(image_bgr, label, (bx + width + 6, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
