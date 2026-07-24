"""Live webcam source (development camera until the D435 arrives)."""
from __future__ import annotations

import time
from typing import Optional

import cv2

from ..log import get_logger
from .base import CameraSource, Frame

logger = get_logger(__name__)


class WebcamSource(CameraSource):
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._req_fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._i = 0

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self._req_fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open webcam index {self.index}")
        logger.info("webcam %d opened (%dx%d)", self.index, self.width, self.height)

    def read(self) -> Optional[Frame]:
        assert self.cap is not None
        ok, color = self.cap.read()
        if not ok:
            return None
        frame = Frame(color=color, index=self._i, timestamp=time.perf_counter())
        self._i += 1
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def fps(self) -> float:
        if self.cap is None:
            return float(self._req_fps)
        return self.cap.get(cv2.CAP_PROP_FPS) or float(self._req_fps)
