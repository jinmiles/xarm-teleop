"""Phase 2 orchestration: camera -> WiLoR -> retarget -> MuJoCo xArm7, rendered side-by-side.

Drives the sim from any CameraSource (webcam or recorded video), so retargeting can be tuned
with no hardware. Clutch auto-engages on the first tracked frame; re-acquisition after a loss
re-engages at the current robot pose (re-indexing).
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..camera import open_source
from ..log import get_logger
from ..perception.realtime import HandTracker
from ..perception.vis import draw_hud, draw_hands, highlight_hand
from ..perception.wilor_estimator import WiLoREstimator
from ..retarget import Retargeter, pinch_to_closed
from .mujoco_env import XArm7Sim

logger = get_logger(__name__)


def _fit_h(img: np.ndarray, h: int) -> np.ndarray:
    s = h / img.shape[0]
    return cv2.resize(img, (round(img.shape[1] * s), h))


def run_sim(
    source: str | int,
    record: Optional[str] = None,
    max_frames: Optional[int] = None,
    scale: float = 3.0,
    depth_scale: float = 0.4,
    pos_only: bool = False,
    primary: str = "auto",
    min_cutoff: float = 1.0,
    beta: float = 0.02,
    proc_max_side: Optional[int] = 640,
    device: Optional[str] = None,
    dtype: str = "float16",
) -> dict:
    est = WiLoREstimator(device=device, dtype=dtype, proc_max_side=proc_max_side, primary=primary)
    tracker = HandTracker(est, min_cutoff=min_cutoff, beta=beta)
    retarget = Retargeter(scale=scale, depth_scale=depth_scale, pos_only=pos_only)
    sim = XArm7Sim(render=True, width=640, height=480)

    writer: Optional[cv2.VideoWriter] = None
    track_err: list[float] = []
    n_frames = n_tracked = 0

    with open_source(source) as cam:
        out_fps = cam.fps or 30.0
        wall = time.perf_counter()
        for frame in cam.frames():
            n_frames += 1
            hands, prim = tracker.update(frame.color, frame.timestamp)

            closed = 0.0
            if prim is not None:
                n_tracked += 1
                if not retarget.engaged:
                    retarget.engage(prim, *sim.tcp_pose())
                tgt_pos, tgt_rot = retarget.target(prim)
                closed = pinch_to_closed(prim.pinch_dist)
                sim.servo_to(tgt_pos, tgt_rot, gripper_closed=closed)
                reached, _ = sim.tcp_pose()
                track_err.append(float(np.linalg.norm(tgt_pos - reached)))
            else:
                sim.hold()

            # compose: camera overlay (left) | sim (right)
            cam_vis = draw_hands(frame.color, hands)
            hud = [f"frame {frame.index}  {'ENGAGED' if retarget.engaged else 'idle'}"]
            if prim is not None:
                highlight_hand(cam_vis, prim)
                hud.append(f"pinch {prim.pinch_dist*1000:.0f}mm -> grip {closed:.2f}")
            draw_hud(cam_vis, hud)
            sim_bgr = cv2.cvtColor(sim.render(), cv2.COLOR_RGB2BGR)
            tcp, _ = sim.tcp_pose()
            draw_hud(sim_bgr, [f"TCP {tcp[0]:+.2f},{tcp[1]:+.2f},{tcp[2]:+.2f}",
                              f"gripper {'closed' if closed>0.5 else 'open'}"])
            composed = np.hstack([_fit_h(cam_vis, 480), _fit_h(sim_bgr, 480)])

            if record:
                if writer is None:
                    Path(record).parent.mkdir(parents=True, exist_ok=True)
                    h, w = composed.shape[:2]
                    writer = cv2.VideoWriter(record, cv2.VideoWriter_fourcc(*"mp4v"),
                                             out_fps, (w, h))
                writer.write(composed)
            if max_frames is not None and frame.index + 1 >= max_frames:
                break
        wall = time.perf_counter() - wall

    if writer is not None:
        writer.release()
    sim.close()

    stats = {
        "frames": n_frames,
        "tracked_ratio": n_tracked / n_frames if n_frames else 0.0,
        "tcp_track_err_mm_median": (statistics.median(track_err) * 1000 if track_err else 0.0),
        "throughput_fps": n_frames / wall if wall > 0 else 0.0,
    }
    logger.info(
        "sim teleop: %d frames | tracked %.0f%% | TCP track err med %.1fmm | %.1f fps",
        stats["frames"], stats["tracked_ratio"] * 100, stats["tcp_track_err_mm_median"],
        stats["throughput_fps"],
    )
    if record:
        logger.info("side-by-side video saved: %s", record)
    return stats
