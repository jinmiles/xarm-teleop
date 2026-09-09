"""Receiver for the Noitom mocap glove stream (Axis Studio -> this PC over the LAN).

Axis Studio runs on the Windows laptop the glove hub is paired with and broadcasts BVH over the
ethernet link; this client is the UDP/TCP peer on the Ubuntu side. It wraps MocapApi through the
external ``mocap_ros_py`` wrapper (see ``paths.import_mocap_api``) and keeps only what the teleop
loop needs: the latest per-bone local rotation and bone translation of one or both hands.

The stream is drained by a background thread so the vision loop never blocks on it and always
reads the freshest frame: the glove runs faster than the camera pipeline, so polling in step with
the camera would consume a backlog of stale frames.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .. import paths
from ..log import get_logger
from .skeleton import SIDE_PREFIX, hand_bones, quat_to_matrix, wrist_bone

logger = get_logger(__name__)

DEFAULT_UDP_PORT = 7012   # must equal the Destination port set in Axis Studio BVH broadcasting


@dataclass
class GloveFrame:
    """One hand's bone rotations from a single glove frame."""

    t: float                                                   # perf_counter when received
    rotations: dict[str, np.ndarray] = field(default_factory=dict)  # bone -> (w,x,y,z) local
    offsets: dict[str, np.ndarray] = field(default_factory=dict)    # bone -> (3,) local, parent frame
    root_rot: np.ndarray = field(default_factory=lambda: np.eye(3))  # wrist bone, mocap world frame


class GloveClient:
    """Latest-frame view of the glove stream. Start with :meth:`connect`, read with :meth:`latest`."""

    def __init__(
        self,
        sides: Iterable[str] = ("right",),
        port: int = DEFAULT_UDP_PORT,
        host: Optional[str] = None,
        poll_interval: float = 0.002,
    ) -> None:
        self.sides = tuple(s.lower() for s in sides)
        for side in self.sides:
            if side not in SIDE_PREFIX:
                raise ValueError(f"glove side must be one of {tuple(SIDE_PREFIX)}: {side!r}")
        self.port = int(port)
        self.host = host          # set for a TCP connection to Axis Studio, else UDP listen
        self.poll_interval = float(poll_interval)

        # bone name -> side, so one traversal of the avatar fills every requested hand
        self._bone_side = {bone: side for side in self.sides
                           for bone in hand_bones(SIDE_PREFIX[side])}
        self._wrist = {wrist_bone(SIDE_PREFIX[side]): side for side in self.sides}

        self._app = None
        self._api = None
        self._settings = None
        self._lock = threading.Lock()
        self._frames: dict[str, GloveFrame] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._count = 0
        self._t_first: Optional[float] = None
        self._warned_bones = False

    # --- lifecycle ---------------------------------------------------------------------
    def connect(self) -> None:
        api = paths.import_mocap_api()
        self._api = api
        settings = api.MCPSettings()
        if self.host:
            settings.set_tcp(self.host, self.port)
        else:
            settings.set_udp(self.port)
        settings.set_bvh_rotation(0)
        app = api.MCPApplication()
        app.set_settings(settings)
        ok, msg = app.open()
        if not ok:
            raise RuntimeError(f"cannot open the glove receiver ({self.where()}): {msg}")
        self._settings = settings   # MCPSettings destroys its handle when garbage collected
        self._app = app
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="glove-rx", daemon=True)
        self._thread.start()
        logger.info("glove receiver listening on %s for %s hand(s)",
                    self.where(), ", ".join(self.sides))

    def where(self) -> str:
        return f"tcp {self.host}:{self.port}" if self.host else f"udp :{self.port}"

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._app is not None:
            self._app.close()
            self._app = None

    def __enter__(self) -> "GloveClient":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- reading -----------------------------------------------------------------------
    def latest(self, side: str) -> Optional[GloveFrame]:
        with self._lock:
            return self._frames.get(side.lower())

    def age(self, side: str) -> Optional[float]:
        """Seconds since the last frame of this hand, or None if none has arrived."""
        frame = self.latest(side)
        return None if frame is None else time.perf_counter() - frame.t

    @property
    def frames_received(self) -> int:
        with self._lock:
            return self._count

    @property
    def hz(self) -> float:
        with self._lock:
            if self._t_first is None or self._count < 2:
                return 0.0
            dt = time.perf_counter() - self._t_first
            return self._count / dt if dt > 0 else 0.0

    def wait_for_data(self, timeout: float = 5.0) -> bool:
        """Block until every requested hand has been seen once. False on timeout."""
        deadline = time.perf_counter() + float(timeout)
        while time.perf_counter() < deadline:
            with self._lock:
                if all(side in self._frames for side in self.sides):
                    return True
            time.sleep(0.02)
        return False

    def require_data(self, timeout: float = 5.0) -> None:
        """Same as :meth:`wait_for_data` but raises with what to check on the Windows side."""
        if self.wait_for_data(timeout):
            logger.info("glove stream live: %.0f frames in, %.0f Hz", self._count, self.hz)
            return
        seen = ", ".join(sorted(self._frames)) or "none"
        raise RuntimeError(
            f"no glove data on {self.where()} after {timeout:.0f}s (hands seen: {seen}). "
            "Check that Axis Studio is streaming (BVH Broadcasting enabled, Destination = this "
            "PC's LAN address and this port, Local = the Windows LAN address, not 127.0.0.1), "
            "that the ethernet link is up, and that the requested hand is actually gloved."
        )

    # --- receive thread ----------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                avatar = self._newest_avatar()
                if avatar is not None:
                    self._store(avatar)
            except Exception:  # a hiccup in the stream must not kill the receiver
                logger.exception("glove receive error (continuing)")
            time.sleep(self.poll_interval)

    def _newest_avatar(self):
        """Drain queued events and return the most recent avatar update, if any."""
        handle = None
        for evt in self._app.poll_next_event():
            if evt.event_type == self._api.MCPEventType.AvatarUpdated:
                handle = evt.event_data.avatar_handle
        return None if handle is None else self._api.MCPAvatar(handle)

    def _store(self, avatar) -> None:
        now = time.perf_counter()
        frames = {side: GloveFrame(t=now) for side in self.sides}
        self._walk(avatar.get_root_joint(), np.eye(3), frames)
        complete = [side for side, f in frames.items() if f.rotations]
        if not complete:
            if not self._warned_bones:
                logger.warning("glove frames carry no %s-hand bones; check the glove side and "
                               "that Axis Studio streams finger data", "/".join(self.sides))
                self._warned_bones = True
            return
        with self._lock:
            for side in complete:
                self._frames[side] = frames[side]
            self._count += 1
            if self._t_first is None:
                self._t_first = now

    def _walk(self, joint, parent_rot: np.ndarray, frames: dict[str, GloveFrame]) -> None:
        """Depth-first over the avatar, accumulating global rotation down to the wrist bones."""
        name = joint.get_name()
        quat = np.asarray(joint.get_local_rotation(), dtype=float)   # (w, x, y, z)
        rot = parent_rot @ quat_to_matrix(quat)
        side = self._bone_side.get(name)
        if side is not None:
            frame = frames[side]
            frame.rotations[name] = quat
            offset = joint.get_local_position()
            if offset is not None:
                frame.offsets[name] = np.asarray(offset, dtype=float)
            if name in self._wrist:
                frame.root_rot = rot
        for child in joint.get_children():
            self._walk(child, rot, frames)
