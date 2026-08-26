"""End-effector interface for hands that are wired independently of the arm.

The UFACTORY 2-finger gripper is driven through the arm controller itself (a scalar in
``RobotBackend.servo_to``), but a dexterous hand such as the Inspire RH56 hangs off its own
serial link. The teleop loop therefore talks to it through this small interface, which keeps the
loop backend-agnostic: sim arm + real hand is a valid (and safe) bring-up combination.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EndEffector(Protocol):
    def connect(self) -> None: ...

    def apply(self, closed: np.ndarray) -> None:
        """Command the effector from per-DOF closed-ratios in [0,1] (1 = fully closed)."""

    def open_hand(self) -> None:
        """Move to the fully open pose (used as a known start state after connect)."""

    def close(self) -> None: ...
