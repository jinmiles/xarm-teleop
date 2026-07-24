"""Real-time hand tracking loop: camera -> WiLoR -> primary-hand selection -> One-Euro filter.

Produces a smoothed 6DOF + pinch stream for a single controlling hand (what teleop needs) and,
optionally, an annotated video / live window plus throughput and latency statistics.
"""
from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..camera import open_source
from ..control.filters import OneEuroFilter
from ..log import get_logger
from .vis import draw_hands, draw_hud, highlight_hand
from .wilor_estimator import HandObservation, WiLoREstimator

logger = get_logger(__name__)


def _bbox_area(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class HandTracker:
    """Selects one controlling hand per frame and smooths its wrist position + pinch."""

    def __init__(
        self,
        estimator: WiLoREstimator,
        primary: str = "auto",           # "auto" | "left" | "right"
        min_cutoff: float = 1.0,
        beta: float = 0.02,
    ) -> None:
        self.est = estimator
        self.primary = primary
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._pos: Optional[OneEuroFilter] = None
        self._pinch: Optional[OneEuroFilter] = None
        self._reset()

    def _reset(self) -> None:
        self._pos = OneEuroFilter(self.min_cutoff, self.beta)
        self._pinch = OneEuroFilter(self.min_cutoff, self.beta)

    def _select(self, hands: list[HandObservation]) -> Optional[HandObservation]:
        cands = hands
        if self.primary == "right":
            cands = [h for h in hands if h.is_right] or hands
        elif self.primary == "left":
            cands = [h for h in hands if not h.is_right] or hands
        if not cands:
            return None
        return max(cands, key=lambda h: _bbox_area(h.bbox))

    def update(self, color_bgr: np.ndarray, t: float):
        """Return (all_hands, filtered_primary_or_None). Filter resets when the hand is lost."""
        hands = self.est.predict(color_bgr)
        sel = self._select(hands)
        if sel is None:
            self._reset()
            return hands, None
        pos = self._pos(sel.wrist_pos_cam, t)
        pinch = float(self._pinch(np.array([sel.pinch_dist]), t)[0])
        return hands, replace(sel, wrist_pos_cam=pos, pinch_dist=pinch)


def run_live(
    source: str | int,
    record: Optional[str] = None,
    display: bool = False,
    max_frames: Optional[int] = None,
    min_cutoff: float = 1.0,
    beta: float = 0.02,
    primary: str = "auto",
    device: Optional[str] = None,
    dtype: str = "float16",
    proc_max_side: Optional[int] = 640,
) -> dict:
    est = WiLoREstimator(device=device, dtype=dtype, proc_max_side=proc_max_side)
    tracker = HandTracker(est, primary=primary, min_cutoff=min_cutoff, beta=beta)

    writer: Optional[cv2.VideoWriter] = None
    recent = deque(maxlen=15)
    infer_ms: list[float] = []
    n_lost = 0

    with open_source(source) as cam:
        out_fps = cam.fps or 30.0
        wall_start = time.perf_counter()
        for frame in cam.frames():
            t0 = time.perf_counter()
            hands, prim = tracker.update(frame.color, frame.timestamp)
            dt_infer = (time.perf_counter() - t0) * 1000.0
            infer_ms.append(dt_infer)
            recent.append(dt_infer)

            vis = draw_hands(frame.color, hands)
            live_fps = 1000.0 / statistics.mean(recent) if recent else 0.0
            hud = [f"fps {live_fps:4.1f}  infer {dt_infer:4.1f}ms  frame {frame.index}"]
            if prim is not None:
                highlight_hand(vis, prim)
                wp = prim.wrist_pos_cam
                hud.append(
                    f"{'R' if prim.is_right else 'L'}  pinch {prim.pinch_dist * 1000:5.1f}mm  "
                    f"wrist {wp[0]:+.2f},{wp[1]:+.2f},{wp[2]:+.2f}m"
                )
            else:
                n_lost += 1
                hud.append("no hand (tracking hold)")
            draw_hud(vis, hud)

            if record:
                if writer is None:
                    Path(record).parent.mkdir(parents=True, exist_ok=True)
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(
                        record, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h)
                    )
                writer.write(vis)
            if display:
                cv2.imshow("xarm-teleop | perception", vis)
                if cv2.waitKey(1) & 0xFF == 27:  # Esc
                    break
            if max_frames is not None and frame.index + 1 >= max_frames:
                break
        wall = time.perf_counter() - wall_start

    if writer is not None:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    n = len(infer_ms)
    stats = {
        "frames": n,
        "throughput_fps": n / wall if wall > 0 else 0.0,
        "infer_ms_mean": statistics.mean(infer_ms) if n else 0.0,
        "infer_ms_median": statistics.median(infer_ms) if n else 0.0,
        "infer_ms_p95": (sorted(infer_ms)[int(0.95 * (n - 1))] if n else 0.0),
        "tracked_ratio": (n - n_lost) / n if n else 0.0,
    }
    logger.info(
        "processed %d frames in %.2fs | throughput %.1f fps | infer med %.1fms p95 %.1fms | "
        "tracked %.0f%%",
        n, wall, stats["throughput_fps"], stats["infer_ms_median"],
        stats["infer_ms_p95"], stats["tracked_ratio"] * 100.0,
    )
    if record:
        logger.info("annotated video saved: %s", record)
    return stats
