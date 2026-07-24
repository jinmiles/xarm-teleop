"""Robot backend interface shared by the MuJoCo sim (Phase 2) and the real xArm7 (Phase 3).

The teleop loop is written against this interface only, so the exact same retargeting, safety
limiting, and control loop drive either the simulator or the hardware. Poses are expressed as
(position in meters, 3x3 rotation matrix) in the robot base frame.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RobotBackend(Protocol):
    def connect(self) -> None: ...
    def home(self) -> None: ...
    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (position[3] meters, rotation[3,3]) of the TCP in the base frame."""

    def servo_to(
        self, target_pos: np.ndarray, target_rot: np.ndarray, gripper_closed: float = 0.0
    ) -> None:
        """Command the TCP toward the target pose (one control tick). gripper_closed in [0,1]."""

    def hold(self) -> None:
        """Keep the last commanded target (called when hand tracking is lost)."""

    def render(self) -> Optional[np.ndarray]:
        """Return an RGB frame of the robot (sim only), or None if not renderable."""

    def close(self) -> None: ...
