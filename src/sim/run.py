"""Phase 2 entry: build the MuJoCo sim backend and drive it with the shared teleop loop."""
from __future__ import annotations

from typing import Optional

from ..control.safety import SafetyLimiter
from ..loop import run_teleop
from ..retarget import Retargeter
from .mujoco_env import XArm7Sim


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
    backend = XArm7Sim(render=True, width=640, height=480)
    retarget = Retargeter(scale=scale, depth_scale=depth_scale, pos_only=pos_only)
    return run_teleop(
        backend=backend,
        source=source,
        retarget=retarget,
        safety=SafetyLimiter(),
        record=record,
        max_frames=max_frames,
        primary=primary,
        min_cutoff=min_cutoff,
        beta=beta,
        proc_max_side=proc_max_side,
        device=device,
        dtype=dtype,
    )
