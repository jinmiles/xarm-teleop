"""Video-file source. Frame timestamps use media time (index / fps) so temporal filters see
the true motion cadence regardless of how fast the model processes the file."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2

from ..log import get_logger
from .base import CameraSource, Frame

logger = get_logger(__name__)


class VideoSource(CameraSource):
    def __init__(self, path: str | Path, loop: bool = False) -> None:
        self.path = Path(path)
        self.loop = loop
        self.cap: Optional[cv2.VideoCapture] = None
        self._fps = 30.0
        self._i = 0

    def open(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video {self.path}")
        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("video %s opened (%.1f fps, %d frames)", self.path.name, self._fps, n)

    def read(self) -> Optional[Frame]:
        assert self.cap is not None
        ok, color = self.cap.read()
        if not ok:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, color = self.cap.read()
            if not ok:
                return None
        frame = Frame(color=color, index=self._i, timestamp=self._i / self._fps)
        self._i += 1
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def fps(self) -> float:
        return self._fps
