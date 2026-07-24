"""Safety limiter: the enforced guard between retargeting and the robot backend.

Independent of retargeting (defense in depth). Clamps every commanded target to a reachable
workspace box and bounds the per-tick TCP translation/rotation (i.e. TCP speed), and provides a
software E-stop that freezes motion. Critical on hardware where an unfiltered target could fling
the arm; harmless in sim but kept in the loop so behavior matches.
"""
from __future__ import annotations

import cv2
import numpy as np

# Conservative reachable box (meters, robot base frame). Tune per cell before hardware use.
DEFAULT_WORKSPACE = np.array([[0.20, 0.65], [-0.40, 0.40], [0.10, 0.65]])


class SafetyLimiter:
    def __init__(
        self,
        workspace: np.ndarray | None = None,
        max_step_m: float = 0.02,     # max TCP translation per control tick (bounds speed)
        max_step_rad: float = 0.10,   # max TCP rotation per control tick
    ) -> None:
        self.workspace = DEFAULT_WORKSPACE if workspace is None else np.asarray(workspace, float)
        self.max_step_m = max_step_m
        self.max_step_rad = max_step_rad
        self.estopped = False

    def estop(self) -> None:
        self.estopped = True

    def clear(self) -> None:
        self.estopped = False

    def limit(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        cur_pos: np.ndarray,
        cur_rot: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a (pos, rot) that is inside the box and within one step of the current pose."""
        if self.estopped:
            return cur_pos.copy(), cur_rot.copy()

        # 1) hard workspace clamp
        pos = np.clip(target_pos, self.workspace[:, 0], self.workspace[:, 1])

        # 2) translation step limit (bounds linear speed)
        dp = pos - cur_pos
        n = float(np.linalg.norm(dp))
        if n > self.max_step_m:
            dp *= self.max_step_m / n
        pos = cur_pos + dp

        # 3) rotation step limit (bounds angular speed)
        r_err = cv2.Rodrigues(target_rot @ cur_rot.T)[0].reshape(3)
        ang = float(np.linalg.norm(r_err))
        if ang > self.max_step_rad:
            r_err *= self.max_step_rad / ang
            rot = cv2.Rodrigues(r_err)[0] @ cur_rot
        else:
            rot = target_rot
        return pos, rot
