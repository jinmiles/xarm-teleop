"""H.264 mp4 writer for browser/VSCode-playable output.

The bundled OpenCV has no H.264 encoder (only mp4v / MPEG-4 Part 2, which HTML5 players such as
VSCode's preview cannot decode), so we pipe raw BGR frames to ffmpeg. Which H.264 encoder exists
depends on the ffmpeg build: GPL builds ship libx264, while LGPL builds (e.g. current conda-forge
ffmpeg) ship only libopenh264 and reject x264-only flags such as `-preset` ("Unrecognized option
'preset'"). The encoder is therefore probed once at runtime and its rate-control flags chosen to
match. Falls back to cv2 mp4v only if no ffmpeg H.264 encoder works.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import IO, Optional

import numpy as np

from .log import get_logger

logger = get_logger(__name__)

# preference order: libx264 (best quality knobs) -> nvenc (GPU) -> libopenh264 (LGPL builds)
_H264_ENCODERS = ("libx264", "h264_nvenc", "libopenh264")
_ENCODER_CACHE: dict[str, Optional[str]] = {}


def _rate_args(encoder: str, crf: int, w: int, h: int, fps: float) -> list[str]:
    if encoder == "libx264":
        return ["-preset", "veryfast", "-crf", str(crf)]
    if encoder == "h264_nvenc":
        return ["-preset", "p4", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    # libopenh264 and friends have no CRF mode: derive a bitrate from frame area x rate
    bitrate = int(min(max(w * h * fps * 0.12, 2e6), 16e6))
    return ["-b:v", str(bitrate)]


def _probe_encoder(ffmpeg: str) -> Optional[str]:
    """First H.264 encoder this ffmpeg can actually run (encode a 2-frame dummy clip)."""
    if ffmpeg in _ENCODER_CACHE:
        return _ENCODER_CACHE[ffmpeg]
    chosen: Optional[str] = None
    for enc in _H264_ENCODERS:
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
               "-i", "color=c=black:s=64x64:r=10:d=0.2", "-c:v", enc,
               "-pix_fmt", "yuv420p", "-f", "null", "-"]
        try:
            done = subprocess.run(cmd, capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0:
            chosen = enc
            break
        logger.debug("ffmpeg encoder %s unusable: %s", enc,
                     done.stderr.decode("utf-8", "replace").strip()[-200:])
    if chosen is None:
        logger.warning("no working H.264 encoder in ffmpeg; falling back to mp4v")
    _ENCODER_CACHE[ffmpeg] = chosen
    return chosen


class VideoWriter:
    def __init__(self, path: str, fps: float, size: tuple[int, int], crf: int = 20) -> None:
        self.path = str(path)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.size = (int(size[0]), int(size[1]))  # (width, height)
        self.crf = crf
        self._proc: subprocess.Popen | None = None
        self._log: IO[bytes] | None = None
        self._cv2 = None
        self._ffmpeg = shutil.which("ffmpeg")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._open()

    def _open(self) -> None:
        w, h = self.size
        encoder = _probe_encoder(self._ffmpeg) if self._ffmpeg else None
        if encoder is None:
            self._open_cv2()
            return
        cmd = [
            self._ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{self.fps}", "-i", "-",
            # pad odd dimensions to even (yuv420p / H.264 requires even w,h)
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", encoder, "-pix_fmt", "yuv420p",
            *_rate_args(encoder, self.crf, w, h, self.fps),
            "-movflags", "+faststart", self.path,
        ]
        # stderr to a temp file, not a pipe: nothing drains a pipe while we stream frames
        self._log = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._log)
        logger.info("recording %dx%d @%.1f with ffmpeg %s", w, h, self.fps, encoder)

    def _open_cv2(self) -> None:
        import cv2
        if self._ffmpeg is None:
            logger.warning("ffmpeg not found; falling back to mp4v (may not play in VSCode)")
        self._cv2 = cv2.VideoWriter(
            self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)

    def _stderr_tail(self) -> str:
        if self._log is None:
            return ""
        try:
            self._log.seek(0)
            return self._log.read().decode("utf-8", "replace").strip()[-500:]
        except OSError:
            return ""

    def _drop_ffmpeg(self, why: str) -> None:
        """ffmpeg died mid-stream: keep the teleop session alive on the cv2 fallback."""
        logger.warning("ffmpeg encoder failed (%s); falling back to mp4v: %s",
                       why, self._stderr_tail() or "no stderr output")
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.kill()
            proc.wait()
        if self._log is not None:
            self._log.close()
            self._log = None
        self._open_cv2()

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._proc is not None:
            assert self._proc.stdin is not None
            try:
                self._proc.stdin.write(np.ascontiguousarray(frame_bgr, dtype=np.uint8).tobytes())
                return
            except (BrokenPipeError, ValueError):
                self._drop_ffmpeg("broken pipe")
        if self._cv2 is not None:
            self._cv2.write(frame_bgr)

    def close(self) -> None:
        if self._proc is not None:
            assert self._proc.stdin is not None
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
            code = self._proc.wait()
            if code != 0:
                logger.error("ffmpeg exited with code %d: %s", code,
                             self._stderr_tail() or "no stderr output")
            self._proc = None
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._cv2 is not None:
            self._cv2.release()
            self._cv2 = None
