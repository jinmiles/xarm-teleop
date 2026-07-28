"""Camera sources and a small factory to open one from a CLI spec."""
from __future__ import annotations

from pathlib import Path

from .base import CameraIntrinsics, CameraSource, Frame
from .video import VideoSource
from .webcam import WebcamSource

__all__ = [
    "CameraIntrinsics", "CameraSource", "Frame", "VideoSource", "WebcamSource", "open_source",
]


def open_source(spec: str | int) -> CameraSource:
    """Interpret a source spec.

    - ``"realsense"`` -> Intel RealSense D435 (color + metric depth)
    - an integer (e.g. ``0``) -> webcam index
    - an existing path -> video file
    """
    s = str(spec)
    if s.lower() in ("realsense", "d435", "rs"):
        from .realsense import RealSenseSource
        return RealSenseSource()
    if s.isdigit():
        return WebcamSource(index=int(s))
    if Path(s).exists():
        return VideoSource(s)
    raise ValueError(
        f"unrecognized camera source: {spec!r} (use 'realsense', a webcam index, or a video path)")
