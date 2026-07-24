"""Real xArm7 backend via xArm-Python-SDK (Phase 3).

Uses the axis-angle servo API (set_servo_cartesian_aa / get_position_aa) so orientation is a
rotation vector -- the same representation cv2.Rodrigues gives us -- avoiding roll/pitch/yaw
convention pitfalls. SDK units are mm + radians (we construct XArmAPI with is_radian=True).

Hardware is not available yet, so ``dry_run=True`` (the default) exercises the full control path
without connecting: servo_to updates an internal echo pose that tcp_pose reports back, so the
loop, retargeting, and safety limiting can be validated end-to-end. Flip to dry_run=False (via
``--execute``) only when a real arm + IP are present, after bring-up review.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..log import get_logger

logger = get_logger(__name__)

# Home joint angles (rad), matching the MuJoCo model's 'home' keyframe.
HOME_Q = np.array([0.0, -0.247, 0.0, 0.909, 0.0, 1.156, 0.0])
# Approx home TCP pose used only for the dry-run echo (real path reads it from the arm).
DRY_HOME_POS = np.array([0.40, 0.0, 0.35])
DRY_HOME_ROT = cv2.Rodrigues(np.array([np.pi, 0.0, 0.0]))[0]  # tool pointing down
GRIPPER_MAX = 850  # UFACTORY gripper: 0 = closed, 850 = open (verify direction on hardware)


class XArm7Controller:
    def __init__(
        self,
        ip: str,
        dry_run: bool = True,
        tcp_speed: float = 100.0,     # mm/s cap for servo moves (kept low for bring-up)
        gripper_speed: int = 2000,
        collision_sensitivity: int = 3,
    ) -> None:
        self.ip = ip
        self.dry_run = dry_run
        self.tcp_speed = tcp_speed
        self.gripper_speed = gripper_speed
        self.collision_sensitivity = collision_sensitivity
        self.arm = None
        self._echo_pos = DRY_HOME_POS.copy()
        self._echo_rot = DRY_HOME_ROT.copy()

    def connect(self) -> None:
        if self.dry_run:
            logger.warning("XArm7Controller dry-run: not connecting to %s (no hardware)", self.ip)
            return
        from xarm.wrapper import XArmAPI

        logger.info("connecting to xArm7 at %s ...", self.ip)
        self.arm = XArmAPI(self.ip, is_radian=True)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self.arm.set_collision_sensitivity(self.collision_sensitivity)
        # gripper
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_speed(self.gripper_speed)
        logger.info("xArm7 connected")

    def home(self) -> None:
        if self.dry_run:
            self._echo_pos, self._echo_rot = DRY_HOME_POS.copy(), DRY_HOME_ROT.copy()
            return
        # move to home in position mode, then switch to Cartesian servo mode (mode 1)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.set_servo_angle(angle=HOME_Q.tolist(), speed=0.35, wait=True, is_radian=True)
        self.arm.set_mode(1)
        self.arm.set_state(0)

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self.dry_run:
            return self._echo_pos.copy(), self._echo_rot.copy()
        code, pose = self.arm.get_position_aa(is_radian=True)  # [x,y,z mm, rx,ry,rz rad]
        pos = np.asarray(pose[:3], dtype=float) / 1000.0
        rot = cv2.Rodrigues(np.asarray(pose[3:6], dtype=float))[0]
        return pos, rot

    def servo_to(
        self, target_pos: np.ndarray, target_rot: np.ndarray, gripper_closed: float = 0.0
    ) -> None:
        rvec = cv2.Rodrigues(target_rot)[0].reshape(3)
        if self.dry_run:
            self._echo_pos, self._echo_rot = np.asarray(target_pos, float).copy(), target_rot.copy()
            return
        pose = [target_pos[0] * 1000.0, target_pos[1] * 1000.0, target_pos[2] * 1000.0,
                float(rvec[0]), float(rvec[1]), float(rvec[2])]
        self.arm.set_servo_cartesian_aa(pose, speed=self.tcp_speed, is_radian=True)
        grip = int((1.0 - float(np.clip(gripper_closed, 0.0, 1.0))) * GRIPPER_MAX)
        self.arm.set_gripper_position(grip, wait=False)

    def hold(self) -> None:
        # In Cartesian servo mode the arm holds its last commanded pose; nothing to do.
        pass

    def render(self) -> Optional[np.ndarray]:
        return None

    def close(self) -> None:
        if self.arm is not None:
            self.arm.set_state(4)  # stop
            self.arm.disconnect()
            self.arm = None
