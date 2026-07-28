"""Backend-agnostic teleop loop: camera -> WiLoR -> retarget -> safety -> robot backend.

Drives either the MuJoCo sim (Phase 2) or the real xArm7 (Phase 3) through the RobotBackend
interface, so the same retargeting, safety limiting, and clutch logic run in both. Renders a
side-by-side view (camera overlay | robot render or status panel) to an H.264 mp4.
"""
from __future__ import annotations

import statistics
import time
from typing import Optional

import cv2
import numpy as np

from .camera import open_source
from .control.backend import RobotBackend
from .control.safety import SafetyLimiter
from .display import PreviewWindow
from .log import get_logger
from .perception.realtime import HandTracker
from .perception.vis import draw_hud, draw_hands, highlight_hand
from .perception.wilor_estimator import WiLoREstimator
from .retarget import Retargeter, pinch_to_closed
from .video import VideoWriter

logger = get_logger(__name__)


def _fit_h(img: np.ndarray, h: int) -> np.ndarray:
    s = h / img.shape[0]
    return cv2.resize(img, (round(img.shape[1] * s), h))


def _status_panel(size: tuple[int, int], lines: list[str]) -> np.ndarray:
    panel = np.full((size[1], size[0], 3), 30, dtype=np.uint8)
    draw_hud(panel, lines)
    return panel


def run_teleop(
    backend: RobotBackend,
    source: str | int,
    retarget: Optional[Retargeter] = None,
    safety: Optional[SafetyLimiter] = None,
    record: Optional[str] = None,
    display: bool = False,
    max_frames: Optional[int] = None,
    primary: str = "auto",
    min_cutoff: float = 1.0,
    beta: float = 0.02,
    proc_max_side: Optional[int] = 640,
    device: Optional[str] = None,
    dtype: str = "float16",
) -> dict:
    est = WiLoREstimator(device=device, dtype=dtype, proc_max_side=proc_max_side, primary=primary)
    tracker = HandTracker(est, min_cutoff=min_cutoff, beta=beta)
    retarget = retarget or Retargeter()
    safety = safety or SafetyLimiter()

    backend.connect()
    backend.home()

    writer: Optional[VideoWriter] = None
    window = PreviewWindow(f"xarm-teleop | {type(backend).__name__}", enabled=display)
    track_err: list[float] = []
    n_frames = n_tracked = 0

    t0 = time.perf_counter()
    try:
        with open_source(source) as cam:
            out_fps = cam.fps or 30.0
            intr = getattr(cam, "intrinsics", None)  # set by depth cameras (D435)
            if intr is not None:
                logger.info("depth camera intrinsics present: wrist will be lifted to metric 3D")
            t0 = time.perf_counter()  # exclude camera warm-up from throughput
            for frame in cam.frames():
                n_frames += 1
                hands, prim = tracker.update(frame.color, frame.timestamp,
                                             depth=frame.depth, intrinsics=intr)
                cur_pos, cur_rot = backend.tcp_pose()

                closed = 0.0
                if prim is not None:
                    n_tracked += 1
                    if not retarget.engaged:
                        retarget.engage(prim, cur_pos, cur_rot)
                    tgt_pos, tgt_rot = retarget.target(prim)
                    closed = pinch_to_closed(prim.pinch_dist)
                    safe_pos, safe_rot = safety.limit(tgt_pos, tgt_rot, cur_pos, cur_rot)
                    backend.servo_to(safe_pos, safe_rot, gripper_closed=closed)
                    reached, _ = backend.tcp_pose()
                    track_err.append(float(np.linalg.norm(safe_pos - reached)))
                else:
                    backend.hold()

                # left: camera overlay
                cam_vis = draw_hands(frame.color, hands)
                hud = [f"frame {frame.index}  {'ENGAGED' if retarget.engaged else 'idle'}"
                       + ("  ESTOP" if safety.estopped else "")]
                if prim is not None:
                    highlight_hand(cam_vis, prim)
                    hud.append(f"pinch {prim.pinch_dist*1000:.0f}mm -> grip {closed:.2f}")
                draw_hud(cam_vis, hud)

                # right: robot render (sim) or status panel (real)
                rgb = backend.render()
                tcp, _ = backend.tcp_pose()
                status = [f"TCP {tcp[0]:+.2f},{tcp[1]:+.2f},{tcp[2]:+.2f}",
                          f"gripper {'closed' if closed > 0.5 else 'open'}"]
                if rgb is not None:
                    right = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    draw_hud(right, status)
                else:
                    right = _status_panel((480, 480), ["[real backend]"] + status)
                composed = np.hstack([_fit_h(cam_vis, 480), _fit_h(right, 480)])

                if record:
                    if writer is None:
                        h, w = composed.shape[:2]
                        writer = VideoWriter(record, out_fps, (w, h))
                    writer.write(composed)
                if not window.show(composed):
                    break
                if max_frames is not None and frame.index + 1 >= max_frames:
                    break
    except KeyboardInterrupt:
        logger.info("interrupted after %d frames; finalizing video and backend", n_frames)
    finally:
        # always finalize: an unfinished mp4 is unplayable and a live arm must leave servo mode
        window.close()
        if writer is not None:
            writer.close()
        backend.close()
    wall = time.perf_counter() - t0

    stats = {
        "frames": n_frames,
        "tracked_ratio": n_tracked / n_frames if n_frames else 0.0,
        "tcp_track_err_mm_median": (statistics.median(track_err) * 1000 if track_err else 0.0),
        "throughput_fps": n_frames / wall if wall > 0 else 0.0,
    }
    logger.info(
        "teleop: %d frames | tracked %.0f%% | TCP track err med %.1fmm | %.1f fps",
        stats["frames"], stats["tracked_ratio"] * 100, stats["tcp_track_err_mm_median"],
        stats["throughput_fps"],
    )
    if record:
        logger.info("side-by-side video saved: %s", record)
    return stats
