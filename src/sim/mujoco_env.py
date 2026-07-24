"""MuJoCo xArm7 environment: Cartesian servoing via differential IK + a parallel-jaw gripper.

Used in Phase 2 to validate retargeting with no hardware risk. The ``servo_to`` loop (damped
least-squares Jacobian step + physics stepping toward the target) mirrors what the real robot
controller will do in Phase 3, so the retargeting tuned here transfers directly.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # headless GPU offscreen rendering

import cv2
import mujoco
import numpy as np

from .. import paths
from ..log import get_logger

logger = get_logger(__name__)

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ARM_ACTS = [f"act{i}" for i in range(1, 8)]


class XArm7Sim:
    def __init__(self, render: bool = True, width: int = 640, height: int = 480) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(paths.XARM7_SCENE))
        self.data = mujoco.MjData(self.model)

        def jid(n):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)

        def aid(n):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

        self.tcp_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "link_tcp")
        self.arm_qadr = np.array([self.model.jnt_qposadr[jid(n)] for n in ARM_JOINTS])
        self.arm_dof = np.array([self.model.jnt_dofadr[jid(n)] for n in ARM_JOINTS])
        self.arm_act = np.array([aid(n) for n in ARM_ACTS])
        self.grip_act = aid("gripper")
        self.jnt_range = np.array([self.model.jnt_range[jid(n)] for n in ARM_JOINTS])  # (7,2)
        self.grip_range = self.model.actuator_ctrlrange[self.grip_act].copy()  # [0,255]

        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

        self.reset_home()

        self.renderer = mujoco.Renderer(self.model, height, width) if render else None
        self.cam = mujoco.MjvCamera()
        self.cam.azimuth, self.cam.elevation, self.cam.distance = 135.0, -20.0, 1.8
        self.cam.lookat[:] = [0.0, 0.0, 0.4]

    # --- RobotBackend interface ----------------------------------------------------------
    def connect(self) -> None:
        pass  # sim needs no connection

    def home(self) -> None:
        self.reset_home()

    # --- state ---------------------------------------------------------------------------
    def reset_home(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # 'home'
        mujoco.mj_forward(self.model, self.data)
        # hold home: position actuators default to ctrl=0, which would drive joints to zero.
        self.data.ctrl[self.arm_act] = self.arm_q()
        self.data.ctrl[self.grip_act] = self.grip_range[0]

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.site_xpos[self.tcp_sid].copy()
        rot = self.data.site_xmat[self.tcp_sid].reshape(3, 3).copy()
        return pos, rot

    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.arm_qadr].copy()

    # --- control -------------------------------------------------------------------------
    def servo_to(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        gripper_closed: float = 0.0,
        damping: float = 0.05,
        max_dq: float = 0.15,
        n_substeps: int = 10,
    ) -> None:
        """One differential-IK step toward the target TCP pose, then advance physics.

        gripper_closed in [0,1] (0 = open, 1 = closed).
        """
        pos, rot = self.tcp_pose()
        e_pos = target_pos - pos
        R_err = target_rot @ rot.T
        e_rot = cv2.Rodrigues(R_err)[0].reshape(3)
        err = np.concatenate([e_pos, e_rot])

        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.tcp_sid)
        J = np.vstack([self._jacp[:, self.arm_dof], self._jacr[:, self.arm_dof]])  # (6,7)
        dq = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(6), err)
        dq = np.clip(dq, -max_dq, max_dq)
        q_target = np.clip(self.arm_q() + dq, self.jnt_range[:, 0], self.jnt_range[:, 1])

        self.data.ctrl[self.arm_act] = q_target
        lo, hi = self.grip_range
        self.data.ctrl[self.grip_act] = lo + float(np.clip(gripper_closed, 0.0, 1.0)) * (hi - lo)
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

    def hold(self, n_substeps: int = 10) -> None:
        """Advance physics holding the current ctrl target (used when tracking is lost)."""
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

    # --- rendering -----------------------------------------------------------------------
    def render(self) -> np.ndarray:
        """Return an RGB frame of the current state."""
        assert self.renderer is not None
        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
