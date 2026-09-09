"""Fuse camera-derived hand position with glove-derived hand pose (teleop method 2).

Method 1 takes everything from the camera: WiLoR gives the wrist position *and* the finger pose.
Method 2 keeps the camera for what only a camera can give -- where the hand is in space -- and
takes the pose from the glove, whose IMUs see finger flexion the camera cannot (self-occlusion,
motion blur, fingers behind the palm) but which knows nothing about position.

The two sensors live in different frames, so the glove pose has to be expressed in the camera
frame before it can drive the same retargeting. The rotation between them is estimated from the
sensors themselves: both observe the same physical hand, so a palm frame built the same way from
each keypoint set differs by exactly that rotation. It is averaged over the first frames after
every (re-)acquisition and then held, which also makes re-indexing the clutch re-align the glove.

Finger retargeting is unaffected by that estimate: joint angles are invariant to a common
rotation of the keypoints, so the fingers work even before the alignment settles.
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Optional

import cv2
import numpy as np

from ..log import get_logger
from ..perception.wilor_estimator import HandObservation
from .client import GloveClient, GloveFrame
from .skeleton import SIDE_PREFIX, keypoints_from_bones, normalize_offsets

logger = get_logger(__name__)

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP = 0, 4, 5, 8, 9


def palm_frame(kp: np.ndarray) -> Optional[np.ndarray]:
    """Orthonormal frame of the palm from 21 keypoints: columns (long axis, in-plane, normal).

    Built identically from either sensor's keypoints, so the same physical palm yields frames
    that differ only by the rotation between the two coordinate systems.
    """
    x = kp[MIDDLE_MCP] - kp[WRIST]
    nx = np.linalg.norm(x)
    if nx < 1e-6:
        return None
    x = x / nx
    z = np.cross(x, kp[INDEX_MCP] - kp[WRIST])
    nz = np.linalg.norm(z)
    if nz < 1e-6:
        return None
    z = z / nz
    return np.stack([x, np.cross(z, x), z], axis=1)


def nearest_rotation(m: np.ndarray) -> np.ndarray:
    """Closest rotation matrix to ``m`` (SVD projection; also averages summed rotations)."""
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:  # reflection -> flip the least-significant axis
        u[:, -1] *= -1.0
        r = u @ vt
    return r


class GloveHandSource:
    """Replaces the pose half of a camera hand observation with the glove's.

    Wrist *position* (and the detection bbox) stay with the camera; keypoints, wrist rotation and
    the pinch distance come from the glove skeleton, rotated into the camera frame.
    """

    def __init__(
        self,
        client: GloveClient,
        side: str = "right",
        timeout: float = 0.3,
        align_frames: int = 30,
    ) -> None:
        self.client = client
        self.side = side.lower()
        self.prefix = SIDE_PREFIX[self.side]
        self.timeout = float(timeout)
        self.align_frames = int(align_frames)
        self._offsets: Optional[dict[str, np.ndarray]] = None
        self._warned_geometry = False
        self._warned_stale = False
        self.reset()

    def reset(self) -> None:
        """Forget the camera<->glove alignment; it is re-estimated on the next frames."""
        self._align = np.eye(3)
        self._align_sum = np.zeros((3, 3))
        self._align_n = 0

    def close(self) -> None:
        self.client.close()

    @property
    def aligned(self) -> bool:
        return self._align_n >= self.align_frames

    # --- glove-only ---------------------------------------------------------------------
    def fresh_frame(self) -> Optional[GloveFrame]:
        """Latest glove frame if it is younger than the timeout, else None.

        Staleness is measured on this machine's clock, not the camera's: a video source carries
        media timestamps, and the glove link has to be judged in real time either way.
        """
        frame = self.client.latest(self.side)
        if frame is None:
            return None
        if time.perf_counter() - frame.t > self.timeout:
            if not self._warned_stale:
                logger.warning("glove frames are older than %.2fs; teleop holds until they resume",
                               self.timeout)
                self._warned_stale = True
            return None
        self._warned_stale = False
        return frame

    def keypoints(self, frame: Optional[GloveFrame] = None) -> Optional[np.ndarray]:
        """(21, 3) MANO keypoints in the mocap world frame, or None if stale/absent."""
        frame = frame if frame is not None else self.fresh_frame()
        if frame is None:
            return None
        if self._offsets is None:
            offsets, complete = normalize_offsets(self.prefix, frame.offsets)
            self._offsets = offsets
            if not complete and not self._warned_geometry:
                logger.warning("glove stream carries no bone lengths; using a nominal hand "
                               "skeleton, so finger angles are approximate")
                self._warned_geometry = True
        kp = keypoints_from_bones(self.prefix, frame.rotations, self._offsets)
        if kp is None:
            return None
        return kp @ frame.root_rot.T   # wrist frame -> mocap world orientation

    def observation(self) -> Optional[HandObservation]:
        """Glove-only observation (no camera): pose is real, position is the origin.

        Used by the calibration and bring-up commands, where only joint angles matter.
        """
        kp = self.keypoints()
        if kp is None:
            return None
        return HandObservation(
            is_right=(self.side == "right"),
            bbox=np.zeros(4),
            wrist_pos_cam=np.zeros(3),
            wrist_rotmat=np.eye(3),
            wrist_aa=np.zeros(3),
            keypoints_3d=kp,
            keypoints_2d=np.zeros((21, 2)),
            pinch_dist=float(np.linalg.norm(kp[THUMB_TIP] - kp[INDEX_TIP])),
            focal_length=0.0,
        )

    # --- fusion -------------------------------------------------------------------------
    def fuse(self, cam: HandObservation, t: float,
             image_shape: Optional[tuple] = None) -> Optional[HandObservation]:
        """Camera position + glove pose, or None when the glove has nothing fresh to say."""
        frame = self.fresh_frame()
        kp_world = self.keypoints(frame)
        if kp_world is None:
            return None
        self._update_align(cam.keypoints_3d, kp_world)

        kp_cam = kp_world @ self._align.T
        rotmat = self._align @ frame.root_rot
        aa, _ = cv2.Rodrigues(rotmat)
        kp2d = cam.keypoints_2d
        if image_shape is not None:
            # Draw the glove hand where the camera hand is, so the overlay shows the pose that
            # actually drives the robot. The focal is fitted from the camera hand's own 2D/3D
            # pair rather than taken from the model: WiLoR reports it for the downscaled frame
            # it ran on, while these pixels are in the full-resolution frame.
            projected = self._project(kp_cam + cam.wrist_pos_cam, cam, image_shape)
            if projected is not None:
                kp2d = projected
        return replace(
            cam,
            is_right=(self.side == "right"),
            keypoints_3d=kp_cam,
            keypoints_2d=kp2d,
            wrist_rotmat=rotmat,
            wrist_aa=aa.reshape(3),
            pinch_dist=float(np.linalg.norm(kp_cam[THUMB_TIP] - kp_cam[INDEX_TIP])),
        )

    def _update_align(self, kp_cam: np.ndarray, kp_world: np.ndarray) -> None:
        if self.aligned:
            return
        cam_frame, glove_frame = palm_frame(kp_cam), palm_frame(kp_world)
        if cam_frame is None or glove_frame is None:
            return
        self._align_sum += cam_frame @ glove_frame.T
        self._align_n += 1
        self._align = nearest_rotation(self._align_sum)
        if self.aligned:
            rpy = cv2.RQDecomp3x3(self._align)[0]  # already degrees
            logger.info("glove->camera alignment locked over %d frames (rpy %.0f,%.0f,%.0f deg)",
                        self._align_n, rpy[0], rpy[1], rpy[2])

    @staticmethod
    def _project(points_cam: np.ndarray, cam: HandObservation,
                 image_shape: tuple) -> Optional[np.ndarray]:
        """Pinhole-project into the camera image, with the focal fitted from ``cam`` itself."""
        h, w = image_shape[:2]
        center = np.array([w / 2.0, h / 2.0])
        ref = cam.keypoints_3d + cam.wrist_pos_cam
        ratio = ref[:, :2] / np.clip(ref[:, 2:3], 1e-4, None)
        denom = float(np.sum(ratio * ratio))
        if denom < 1e-9:
            return None
        focal = float(np.sum(ratio * (cam.keypoints_2d - center))) / denom
        if not np.isfinite(focal) or focal <= 0.0:
            return None
        z = np.clip(points_cam[:, 2:3], 1e-4, None)
        return focal * points_cam[:, :2] / z + center
