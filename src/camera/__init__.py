"""Camera sources and a small factory to open one from a CLI spec."""
from __future__ import annotations

from pathlib import Path

from .base import CameraSource, Frame
from .video import VideoSource
from .webcam import WebcamSource

__all__ = ["CameraSource", "Frame", "VideoSource", "WebcamSource", "open_source"]


def open_source(spec: str | int) -> CameraSource:
    """Interpret a source spec: an integer -> webcam index, an existing path -> video file."""
    s = str(spec)
    if s.isdigit():
        return WebcamSource(index=int(s))
    if Path(s).exists():
        return VideoSource(s)
    raise ValueError(f"unrecognized camera source: {spec!r} (use a webcam index or a video path)")
