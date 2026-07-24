"""WiLoR hand pose estimator wrapper.

Wraps the wilor-mini inference pipeline and exposes a project-owned ``HandObservation`` with
exactly the quantities the teleop pipeline needs: wrist 6DOF (camera frame), pinch distance,
and 2D/3D keypoints. The heavy upstream model is kept behind this shim; project code depends
on ``HandObservation``, not on wilor-mini's raw dict layout.

WiLoR output notes (from wilor_mini pipeline):
  - predict(rgb) -> list of dicts, one per detected hand.
  - dict["is_right"]: 0.0/1.0 ; dict["hand_bbox"]: [x1,y1,x2,y2]
  - dict["wilor_preds"]:
      pred_keypoints_3d (1,21,3)  root-relative meters (MANO wrist at origin)
      pred_keypoints_2d (1,21,2)  image px
      pred_cam_t_full   (1,3)     hand-root translation in camera frame (meters)
      global_orient     (1,1,3)   wrist rotation, axis-angle
      scaled_focal_length scalar
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .. import paths
from ..log import get_logger

logger = get_logger(__name__)

# 21-keypoint hand layout (MediaPipe/OpenPose order used by WiLoR / MANO regressors).
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# Bone connectivity for visualization (parent -> child chains from the wrist).
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
]


@dataclass
class HandObservation:
    """A single detected hand, expressed in the camera frame (meters, right-handed)."""

    is_right: bool
    bbox: np.ndarray            # (4,) x1,y1,x2,y2 in image px
    wrist_pos_cam: np.ndarray   # (3,) camera-frame translation, meters
    wrist_rotmat: np.ndarray    # (3,3) camera-frame rotation
    wrist_aa: np.ndarray        # (3,) axis-angle
    keypoints_3d: np.ndarray    # (21,3) root-relative meters
    keypoints_2d: np.ndarray    # (21,2) image px
    pinch_dist: float           # thumb tip <-> index tip distance, meters
    focal_length: float

    @property
    def pinch_ratio(self) -> float:
        """Crude 0..1 open ratio (0=closed pinch). Retargeting tunes the real mapping later."""
        return float(np.clip(self.pinch_dist / 0.10, 0.0, 1.0))


class WiLoREstimator:
    """Thin wrapper around wilor-mini's 3D hand pose pipeline."""

    def __init__(self, device: Optional[str] = None, dtype: str = "float16") -> None:
        paths.configure_hf_cache()
        import torch  # deferred: torch belongs to the model env, not to import-time of paths
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )

        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        logger.info("loading WiLoR pipeline (device=%s dtype=%s) ...", device, dtype)
        self.pipe = WiLorHandPose3dEstimationPipeline(device=self.device, dtype=self.dtype)
        logger.info("WiLoR pipeline ready")

    def predict(self, image: np.ndarray, is_bgr: bool = True) -> list[HandObservation]:
        """Run WiLoR on one frame. ``image`` is HxWx3 uint8 (BGR by default, as from OpenCV)."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if is_bgr else image
        raw = self.pipe.predict(rgb)
        return [self._parse(d) for d in raw]

    @staticmethod
    def _parse(det: dict) -> HandObservation:
        w = det["wilor_preds"]
        kp3d = np.asarray(w["pred_keypoints_3d"], dtype=float).reshape(-1, 3)  # (21,3)
        kp2d = np.asarray(w["pred_keypoints_2d"], dtype=float).reshape(-1, 2)  # (21,2)
        cam_t = np.asarray(w["pred_cam_t_full"], dtype=float).reshape(3)
        aa = np.asarray(w["global_orient"], dtype=float).reshape(3)
        rotmat, _ = cv2.Rodrigues(aa)
        wrist_pos = cam_t + kp3d[WRIST]
        pinch = float(np.linalg.norm(kp3d[THUMB_TIP] - kp3d[INDEX_TIP]))
        return HandObservation(
            is_right=bool(round(float(det["is_right"]))),
            bbox=np.asarray(det["hand_bbox"], dtype=float).reshape(4),
            wrist_pos_cam=wrist_pos,
            wrist_rotmat=np.asarray(rotmat, dtype=float),
            wrist_aa=aa,
            keypoints_3d=kp3d,
            keypoints_2d=kp2d,
            pinch_dist=pinch,
            focal_length=float(np.asarray(w["scaled_focal_length"]).reshape(-1)[0]),
        )
