"""Real xArm7 backend via xArm-Python-SDK (Phase 3).

Uses the axis-angle servo API (set_servo_cartesian_aa / get_position_aa) so orientation is a
rotation vector -- the same representation cv2.Rodrigues gives us -- avoiding roll/pitch/yaw
convention pitfalls. SDK units are mm + radians (we construct XArmAPI with is_radian=True).

With a dexterous hand mounted in place of the UFACTORY 2-finger gripper, pass
``use_gripper=False``: the controller then never touches the gripper API. Enabling or commanding a
gripper that is not on the tool flange makes the controller latch error 19 ("End Effector
Communication Error"), after which every motion command is rejected with code 1 and the arm stops
dead -- the hand keeps working, so it looks like an arm-only failure.

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

# Controller errors this loop can actually cause, with the fix rather than the vendor wording.
ERROR_HINTS = {
    19: "end effector communication failed -- nothing answers on the tool RS485 bus. Run with "
        "--no-gripper (implied by --hand-port), and in UFACTORY Studio uninstall the end effector "
        "so the controller stops polling for it",
    21: "kinematic error -- the target is unreachable; lower --scale or shrink DEFAULT_WORKSPACE",
    22: "self-collision -- re-home the arm and shrink DEFAULT_WORKSPACE",
    24: "speed limit exceeded -- lower --tcp-speed and --max-step-m",
}


class XArm7Controller:
    def __init__(
        self,
        ip: str,
        dry_run: bool = True,
        tcp_speed: float = 100.0,     # mm/s cap for servo moves (kept low for bring-up)
        gripper_speed: int = 2000,
        collision_sensitivity: int = 3,
        use_gripper: bool = True,
    ) -> None:
        self.ip = ip
        self.dry_run = dry_run
        self.tcp_speed = tcp_speed
        self.gripper_speed = gripper_speed
        self.collision_sensitivity = collision_sensitivity
        self.use_gripper = bool(use_gripper)
        self.arm = None
        self._echo_pos = DRY_HOME_POS.copy()
        self._echo_rot = DRY_HOME_ROT.copy()
        self._n_servo_err = 0

    def connect(self) -> None:
        if self.dry_run:
            logger.warning("XArm7Controller dry-run: not connecting to %s (no hardware)", self.ip)
            return
        from xarm.wrapper import XArmAPI

        logger.info("connecting to xArm7 at %s ...", self.ip)
        # baud_checkset makes the SDK write the gripper's baud rate onto the tool RS485 bus before
        # any gripper call; with nothing on the flange that write is what raises error 19.
        self.arm = XArmAPI(self.ip, is_radian=True, baud_checkset=self.use_gripper)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        if self.arm.error_code:
            raise RuntimeError(
                f"xArm still reports controller error {self.arm.error_code} after clean_error: "
                + ERROR_HINTS.get(self.arm.error_code, "clear it in UFACTORY Studio and retry"))
        self.arm.set_collision_sensitivity(self.collision_sensitivity)
        if self.use_gripper:
            self.arm.set_gripper_enable(True)
            self.arm.set_gripper_mode(0)
            self.arm.set_gripper_speed(self.gripper_speed)
        else:
            logger.info("UFACTORY gripper disabled: the tool flange carries another end effector")
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
        code = self.arm.set_servo_cartesian_aa(pose, speed=self.tcp_speed, is_radian=True)
        self._check_servo(code)
        if self.use_gripper:
            grip = int((1.0 - float(np.clip(gripper_closed, 0.0, 1.0))) * GRIPPER_MAX)
            self.arm.set_gripper_position(grip, wait=False)

    def _check_servo(self, code: int) -> None:
        """Report a rejected servo command once, with the controller error behind it.

        The SDK logs each rejection itself but keeps going, so without this a latched arm error
        looks like a silently frozen robot while the loop happily reports frames per second.
        """
        if not code:
            self._n_servo_err = 0
            return
        self._n_servo_err += 1
        if self._n_servo_err not in (1, 10) and self._n_servo_err % 200:
            return
        err = getattr(self.arm, "error_code", None)
        hint = ERROR_HINTS.get(err, "clear it in UFACTORY Studio, then restart teleop")
        logger.error("arm rejected servo command (code=%s, arm error=%s; %d so far): %s",
                     code, err, self._n_servo_err, hint)

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
