"""Lift the wrist keypoint to metric 3D using the D435 aligned depth + color intrinsics.

WiLoR's translation is only up-to-scale from a monocular image (the camera-Z is unreliable).
With a depth camera we replace the wrist position with a true metric back-projection of the
wrist pixel, so relative teleop deltas become metrically correct (use scale~1, depth_scale~1).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..camera.base import CameraIntrinsics
from .wilor_estimator import WRIST, HandObservation


def backproject(u: float, v: float, z: float, intr: CameraIntrinsics) -> np.ndarray:
    """Pixel (u,v) at depth z (meters) -> 3D point in the camera frame (meters)."""
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    return np.array([x, y, z], dtype=float)


def _sample_depth(depth_m: np.ndarray, u: float, v: float, win: int = 2) -> float:
    """Median of valid depths in a small window around (u,v). 0.0 if none valid."""
    h, w = depth_m.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return 0.0
    patch = depth_m[max(0, vi - win):vi + win + 1, max(0, ui - win):ui + win + 1]
    valid = patch[(patch > 0.05) & (patch < 5.0)]  # ignore zeros / out-of-range
    return float(np.median(valid)) if valid.size else 0.0


def refine_wrist(hand: HandObservation, depth_m: np.ndarray, intr: CameraIntrinsics) -> HandObservation:
    """Return a copy of ``hand`` with wrist_pos_cam replaced by the metric back-projection.

    Falls back to the original (monocular) wrist position if depth at the wrist is invalid.
    """
    u, v = hand.keypoints_2d[WRIST]
    z = _sample_depth(depth_m, u, v)
    if z <= 0.0:
        return hand
    return replace(hand, wrist_pos_cam=backproject(u, v, z, intr))
