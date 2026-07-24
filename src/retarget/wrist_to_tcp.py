"""Relative (clutch-based) retargeting of the human wrist 6DOF to the robot TCP pose.

On clutch engage we snapshot the reference hand pose H0 and robot pose R0; while engaged the
TCP target is R0 plus the scaled, frame-aligned hand displacement. Relative control needs no
camera<->robot calibration and lets the operator re-index (disengage, reposition, re-engage).

Note on depth: a monocular hand model gives an unreliable camera-Z (depth). Until the D435
supplies metric depth (Phase 4), the depth axis is down-weighted via ``depth_scale``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def default_align() -> np.ndarray:
    """Camera frame (x right, y down, z fwd) -> robot base (x fwd, y left, z up)."""
    return np.array(
        [
            [0.0, 0.0, -1.0],   # robot_x <- -cam_z (toward camera => forward)
            [-1.0, 0.0, 0.0],   # robot_y <- -cam_x (hand right => robot right)
            [0.0, -1.0, 0.0],   # robot_z <- -cam_y (hand up => robot up)
        ]
    )


# A conservative reachable box (meters, robot base frame) to keep targets sane in sim/hardware.
DEFAULT_WORKSPACE = np.array([[0.20, 0.70], [-0.40, 0.40], [0.10, 0.70]])


class Retargeter:
    def __init__(
        self,
        scale: float = 3.0,
        depth_scale: float = 0.4,
        R_align: Optional[np.ndarray] = None,
        pos_only: bool = False,
        workspace: Optional[np.ndarray] = None,
    ) -> None:
        self.scale = scale
        self.depth_scale = depth_scale
        self.R_align = default_align() if R_align is None else np.asarray(R_align, dtype=float)
        self.pos_only = pos_only
        self.workspace = DEFAULT_WORKSPACE if workspace is None else np.asarray(workspace, float)
        self.engaged = False
        self.h0_pos = self.h0_rot = self.r0_pos = self.r0_rot = None

    def engage(self, hand, tcp_pos: np.ndarray, tcp_rot: np.ndarray) -> None:
        self.h0_pos = hand.wrist_pos_cam.copy()
        self.h0_rot = hand.wrist_rotmat.copy()
        self.r0_pos = np.asarray(tcp_pos, dtype=float).copy()
        self.r0_rot = np.asarray(tcp_rot, dtype=float).copy()
        self.engaged = True

    def disengage(self) -> None:
        self.engaged = False

    def target(self, hand) -> tuple[np.ndarray, np.ndarray]:
        """Return (target_pos, target_rot) in the robot base frame. Requires ``engaged``."""
        d_cam = hand.wrist_pos_cam - self.h0_pos
        d_cam = d_cam * np.array([self.scale, self.scale, self.depth_scale])
        tgt_pos = self.r0_pos + self.R_align @ d_cam
        tgt_pos = np.clip(tgt_pos, self.workspace[:, 0], self.workspace[:, 1])

        if self.pos_only:
            tgt_rot = self.r0_rot
        else:
            dR_cam = hand.wrist_rotmat @ self.h0_rot.T
            dR_robot = self.R_align @ dR_cam @ self.R_align.T
            tgt_rot = dR_robot @ self.r0_rot
        return tgt_pos, tgt_rot
