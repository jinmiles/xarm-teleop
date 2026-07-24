"""RealSense D435 source (Phase 4). Placeholder until the device is available.

When enabled this streams aligned color + depth so perception can lift the wrist keypoint to
metric 3D. Kept behind the same CameraSource interface as webcam/video.
"""
from __future__ import annotations

from typing import Optional

from .base import CameraSource, Frame


class RealSenseSource(CameraSource):
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self._fps = fps

    def open(self) -> None:
        raise NotImplementedError(
            "RealSenseSource is not enabled yet (D435 not available). "
            "Phase 4: pip install pyrealsense2, connect the D435, then implement aligned "
            "color+depth streaming here."
        )

    def read(self) -> Optional[Frame]:  # pragma: no cover - not implemented
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - not implemented
        pass

    @property
    def fps(self) -> float:
        return float(self._fps)
