"""H.264 mp4 writer for browser/VSCode-playable output.

The bundled OpenCV has no H.264 encoder (only mp4v / MPEG-4 Part 2, which HTML5 players such as
VSCode's preview cannot decode). We pipe raw BGR frames to ffmpeg + libx264 (yuv420p) instead,
which every HTML5 player supports. Falls back to cv2 mp4v only if ffmpeg is unavailable.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from .log import get_logger

logger = get_logger(__name__)


class VideoWriter:
    def __init__(self, path: str, fps: float, size: tuple[int, int], crf: int = 20) -> None:
        self.path = str(path)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.size = (int(size[0]), int(size[1]))  # (width, height)
        self.crf = crf
        self._proc: subprocess.Popen | None = None
        self._cv2 = None
        self._ffmpeg = shutil.which("ffmpeg")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._open()

    def _open(self) -> None:
        w, h = self.size
        if self._ffmpeg:
            cmd = [
                self._ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}", "-r", f"{self.fps}", "-i", "-",
                # pad odd dimensions to even (yuv420p / libx264 requires even w,h)
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                "-crf", str(self.crf), "-movflags", "+faststart", self.path,
            ]
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            import cv2
            logger.warning("ffmpeg not found; falling back to mp4v (may not play in VSCode)")
            self._cv2 = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._proc is not None:
            assert self._proc.stdin is not None
            self._proc.stdin.write(np.ascontiguousarray(frame_bgr, dtype=np.uint8).tobytes())
        elif self._cv2 is not None:
            self._cv2.write(frame_bgr)

    def close(self) -> None:
        if self._proc is not None:
            assert self._proc.stdin is not None
            self._proc.stdin.close()
            self._proc.wait()
            self._proc = None
        if self._cv2 is not None:
            self._cv2.release()
            self._cv2 = None
