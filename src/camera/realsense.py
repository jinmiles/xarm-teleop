"""Intel RealSense D435 source: color + depth aligned to color, with metric depth in meters.

Streams BGR color and a depth map aligned to the color frame, plus the color intrinsics so
perception can back-project the wrist keypoint to true metric 3D (Phase 4). Same CameraSource
interface as webcam/video, so `--source realsense` is a drop-in for the teleop commands.

Requires a physical D435 on USB3 and `pyrealsense2`. Not testable without the device; the API
calls here follow the standard librealsense pattern.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..log import get_logger
from .base import CameraIntrinsics, CameraSource, Frame

logger = get_logger(__name__)


class RealSenseSource(CameraSource):
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self._fps = fps
        self._pipe = None
        self._align = None
        self._depth_scale = 1.0
        self._i = 0

    def open(self) -> None:
        import pyrealsense2 as rs

        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self._fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self._fps)
        profile = self._pipe.start(cfg)

        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()  # raw units -> meters
        self._align = rs.align(rs.stream.color)  # align depth into the color frame

        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = CameraIntrinsics(
            fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy, width=intr.width, height=intr.height
        )
        logger.info("D435 opened (%dx%d @%d, fx=%.1f fy=%.1f depth_scale=%.5f)",
                    self.width, self.height, self._fps, intr.fx, intr.fy, self._depth_scale)

    def read(self) -> Optional[Frame]:
        assert self._pipe is not None and self._align is not None
        frames = self._align.process(self._pipe.wait_for_frames())
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            return None
        color_np = np.asanyarray(color.get_data())  # HxWx3 BGR
        depth_np = np.asanyarray(depth.get_data()).astype(np.float32) * self._depth_scale  # meters
        frame = Frame(color=color_np, index=self._i, timestamp=time.perf_counter(), depth=depth_np)
        self._i += 1
        return frame

    def close(self) -> None:
        if self._pipe is not None:
            self._pipe.stop()
            self._pipe = None

    @property
    def fps(self) -> float:
        return float(self._fps)
