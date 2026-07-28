"""Optional live preview window shared by the perception and teleop loops.

`cv2.imshow` needs a GUI-capable OpenCV build and a reachable display, neither of which is
guaranteed on a robot PC or over SSH. It cannot simply be wrapped in try/except: with no usable
display OpenCV's Qt backend calls qFatal() and aborts the process, which would kill a run that is
commanding a real arm. So GUI support is probed once in a throwaway child process (an abort there
is just a nonzero exit code) and the window turns into a no-op if it fails.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

import cv2
import numpy as np

from .log import get_logger

logger = get_logger(__name__)

_QUIT_KEYS = (27, ord("q"))  # Esc, q
_PROBE = (
    "import cv2, numpy as np; cv2.namedWindow('probe'); "
    "cv2.imshow('probe', np.zeros((4, 4, 3), np.uint8)); cv2.waitKey(1); cv2.destroyAllWindows()"
)
_gui_ok: Optional[bool] = None


def gui_available() -> bool:
    """Whether cv2 windows work here (probed once per process)."""
    global _gui_ok
    if _gui_ok is not None:
        return _gui_ok
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        logger.warning("no DISPLAY/WAYLAND_DISPLAY set; live window disabled (recording continues)")
        _gui_ok = False
        return _gui_ok
    try:
        done = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, timeout=20)
        _gui_ok = done.returncode == 0
        if not _gui_ok:
            logger.warning("opencv cannot open a window here; live window disabled: %s",
                           done.stderr.decode("utf-8", "replace").strip()[-200:] or "aborted")
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("gui probe failed (%s); live window disabled", exc)
        _gui_ok = False
    return _gui_ok


class PreviewWindow:
    def __init__(self, title: str, enabled: bool = True) -> None:
        self.title = title
        self.enabled = bool(enabled) and gui_available()
        self._opened = False

    def show(self, frame_bgr: np.ndarray) -> bool:
        """Draw one frame. Returns False when the user asked to quit (Esc / q)."""
        if not self.enabled:
            return True
        try:
            cv2.imshow(self.title, frame_bgr)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            self.enabled = False
            detail = str(exc).strip().splitlines()[-1] if str(exc).strip() else repr(exc)
            logger.warning("live display failed (%s); continuing headless", detail)
            return True
        self._opened = True
        if key in _QUIT_KEYS:
            logger.info("quit key pressed; stopping")
            return False
        return True

    def close(self) -> None:
        if not self._opened:
            return
        self._opened = False
        try:
            cv2.destroyWindow(self.title)
            cv2.waitKey(1)  # let the GUI event loop process the destroy
        except cv2.error:
            pass
