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

    def __init__(
        self,
        device: Optional[str] = None,
        dtype: str = "float16",
        proc_max_side: Optional[int] = None,
        primary: str = "auto",
        det_conf: float = 0.3,
    ) -> None:
        # Single-hand inference: teleop controls with one hand, so we detect all hands but
        # reconstruct only the selected one. primary: "auto" (largest bbox) | "left" | "right".
        self.primary = primary
        self.det_conf = det_conf
        # proc_max_side: downscale the frame so its longest side <= this before inference.
        # wilor-mini runs a CPU gaussian blur over the *whole frame* per hand when the hand
        # bbox is large (~>563 px), which dominates latency at high resolution. Keeping the
        # processed frame small avoids that trigger (~5x faster at 1080p) with no loss of hand
        # detail (the model crops to 256 px anyway). None = no downscale.
        self.proc_max_side = proc_max_side
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
        logger.info("loading WiLoR pipeline (device=%s dtype=%s proc_max_side=%s primary=%s) ...",
                    device, dtype, proc_max_side, primary)
        self.pipe = WiLorHandPose3dEstimationPipeline(device=self.device, dtype=self.dtype)
        logger.info("WiLoR pipeline ready")

    def predict(self, image: np.ndarray, is_bgr: bool = True) -> list[HandObservation]:
        """Detect hands, then reconstruct only the single controlling hand.

        Returns a list of length 0 (no hand) or 1 (the selected hand), so downstream code has a
        uniform interface. ``image`` is HxWx3 uint8 (BGR by default, as from OpenCV).
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if is_bgr else image
        h, w = rgb.shape[:2]
        scale = 1.0
        proc = rgb
        if self.proc_max_side is not None and max(h, w) > self.proc_max_side:
            scale = self.proc_max_side / float(max(h, w))
            proc = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)

        chosen = self._select(self._detect(proc))
        if chosen is None:
            return []
        bbox, is_right = chosen
        raw = self.pipe.predict_with_bboxes(proc, np.asarray([bbox], dtype=float), [is_right])
        if not raw:
            return []
        return [self._parse(raw[0], kp2d_scale=1.0 / scale)]

    def _detect(self, proc_rgb: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Run only the YOLO hand detector. Returns [(bbox xyxy, is_right), ...]."""
        res = self.pipe.hand_detector(proc_rgb, conf=self.det_conf, verbose=False)[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()  # handedness: 0=left, 1=right
        return [(xyxy[i], float(cls[i])) for i in range(len(xyxy))]

    def _select(self, cands: list[tuple[np.ndarray, float]]) -> Optional[tuple[np.ndarray, float]]:
        if not cands:
            return None
        pool, forced = cands, None
        if self.primary == "right":
            pool, forced = [c for c in cands if round(c[1]) == 1] or cands, 1.0
        elif self.primary == "left":
            pool, forced = [c for c in cands if round(c[1]) == 0] or cands, 0.0
        bbox, is_right = max(pool, key=lambda c: (c[0][2] - c[0][0]) * (c[0][3] - c[0][1]))
        # An explicit --primary is the operator telling us which hand they teleop with, so it wins
        # over the detector's handedness class -- a mislabel there flips the palm normal and with
        # it the thumb-rotation direction, while the four fingers look fine.
        return bbox, (is_right if forced is None else forced)

    @staticmethod
    def _parse(det: dict, kp2d_scale: float = 1.0) -> HandObservation:
        w = det["wilor_preds"]
        kp3d = np.asarray(w["pred_keypoints_3d"], dtype=float).reshape(-1, 3)  # (21,3)
        # 2D outputs are in processed-image pixels; rescale to the original frame resolution.
        kp2d = np.asarray(w["pred_keypoints_2d"], dtype=float).reshape(-1, 2) * kp2d_scale
        cam_t = np.asarray(w["pred_cam_t_full"], dtype=float).reshape(3)
        aa = np.asarray(w["global_orient"], dtype=float).reshape(3)
        rotmat, _ = cv2.Rodrigues(aa)
        wrist_pos = cam_t + kp3d[WRIST]
        pinch = float(np.linalg.norm(kp3d[THUMB_TIP] - kp3d[INDEX_TIP]))
        return HandObservation(
            is_right=bool(round(float(det["is_right"]))),
            bbox=np.asarray(det["hand_bbox"], dtype=float).reshape(4) * kp2d_scale,
            wrist_pos_cam=wrist_pos,
            wrist_rotmat=np.asarray(rotmat, dtype=float),
            wrist_aa=aa,
            keypoints_3d=kp3d,
            keypoints_2d=kp2d,
            pinch_dist=pinch,
            focal_length=float(np.asarray(w["scaled_focal_length"]).reshape(-1)[0]),
        )
