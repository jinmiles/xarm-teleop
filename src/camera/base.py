"""Camera source abstraction.

The rest of the pipeline depends only on ``CameraSource`` / ``Frame``, never on a concrete
device. This lets Phases 0-3 run on a webcam or a recorded video and swap in the RealSense
D435 (with metric depth) in Phase 4 without touching perception or control code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Frame:
    color: np.ndarray                 # HxWx3 uint8, BGR (OpenCV convention)
    index: int                        # 0-based frame counter
    timestamp: float                  # seconds; media time for files, wall time for live
    depth: Optional[np.ndarray] = None  # HxW float32 meters (RealSense only), else None


class CameraSource(ABC):
    """A stream of frames. Use as a context manager; iterate with :meth:`frames`."""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Return the next frame, or ``None`` when the stream ends."""

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def fps(self) -> float: ...

    def frames(self):
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame

    def __enter__(self) -> "CameraSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
