"""Retarget MANO hand keypoints to a 6-DOF dexterous hand (Inspire RH56 layout).

Per-finger curl is the summed flexion of the joints along each finger chain, which is
scale-invariant (angles, not distances) and therefore independent of the operator's hand size --
unlike the pinch distance used for the 2-finger gripper. Thumb rotation (opposition) is measured
as the thumb metacarpal's angle out of the palm plane, so it stays decoupled from thumb flexion.

Raw angles are mapped to closed-ratios in [0,1] through a per-DOF open/closed calibration; the
built-in defaults are approximate, so run ``teleop.py hand-calib`` once per operator and pass the
resulting JSON with ``--hand-calib``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..control.filters import OneEuroFilter
from ..log import get_logger

logger = get_logger(__name__)

WRIST = 0
# (mcp, pip, dip, tip) per finger; the wrist is the chain root
FINGER_CHAINS = {
    "little": (17, 18, 19, 20),
    "ring": (13, 14, 15, 16),
    "middle": (9, 10, 11, 12),
    "index": (5, 6, 7, 8),
}
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, LITTLE_MCP = 5, 17

# Inspire RH56 DOF order
DOF_NAMES = ("little", "ring", "middle", "index", "thumb_bend", "thumb_rot")
N_DOF = 6

# Approximate defaults in radians: 4 finger curls (sum of 3 joints), thumb curl (2 joints),
# thumb out-of-palm angle. Replaced by a real capture from `teleop.py hand-calib`.
DEFAULT_OPEN = np.array([0.30, 0.30, 0.30, 0.30, 0.20, -0.10])
DEFAULT_CLOSED = np.array([3.60, 3.60, 3.60, 3.60, 1.60, 0.90])


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    return float(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)))


def _chain_curl(kp: np.ndarray, chain: tuple[int, ...], root: int = WRIST) -> float:
    """Sum of the flexion angles between consecutive bones along a finger chain."""
    pts = [kp[root]] + [kp[i] for i in chain]
    bones = [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]
    return float(sum(_angle(bones[i], bones[i + 1]) for i in range(len(bones) - 1)))


def palm_normal(kp: np.ndarray, is_right: bool) -> np.ndarray:
    """Unit normal of the palm plane, signed so that thumb opposition is positive."""
    n = np.cross(kp[INDEX_MCP] - kp[WRIST], kp[LITTLE_MCP] - kp[WRIST])
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    n = n / norm
    return n if is_right else -n


@dataclass
class DexCalibration:
    """Per-DOF raw angle (radians) at the fully open and fully closed hand poses."""

    open_rad: np.ndarray
    closed_rad: np.ndarray

    @classmethod
    def default(cls) -> "DexCalibration":
        return cls(DEFAULT_OPEN.copy(), DEFAULT_CLOSED.copy())

    @classmethod
    def load(cls, path: str | Path) -> "DexCalibration":
        blob = json.loads(Path(path).read_text())
        calib = cls(np.asarray(blob["open_rad"], dtype=float),
                    np.asarray(blob["closed_rad"], dtype=float))
        if calib.open_rad.shape != (N_DOF,) or calib.closed_rad.shape != (N_DOF,):
            raise ValueError(f"calibration must hold {N_DOF} values per pose: {path}")
        return calib

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(
            {"dof_names": list(DOF_NAMES),
             "open_rad": [round(v, 4) for v in self.open_rad],
             "closed_rad": [round(v, 4) for v in self.closed_rad]}, indent=2))

    def ratio(self, raw: np.ndarray) -> np.ndarray:
        span = self.closed_rad - self.open_rad
        span = np.where(np.abs(span) < 1e-6, 1e-6, span)
        return np.clip((raw - self.open_rad) / span, 0.0, 1.0)


class DexHandRetargeter:
    """MANO keypoints -> 6 closed-ratios in [0,1] (1 = fully bent / opposed)."""

    def __init__(self, calib: Optional[DexCalibration] = None,
                 min_cutoff: float = 1.5, beta: float = 0.02) -> None:
        self.calib = calib or DexCalibration.default()
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self._filt: Optional[OneEuroFilter] = None

    def reset(self) -> None:
        self._filt = None

    def raw(self, obs) -> np.ndarray:
        """Raw per-DOF angles in radians, in Inspire DOF order (uncalibrated, unsmoothed)."""
        kp = np.asarray(obs.keypoints_3d, dtype=float)
        curls = [_chain_curl(kp, FINGER_CHAINS[name]) for name in DOF_NAMES[:4]]
        thumb_bend = (_angle(kp[THUMB_MCP] - kp[THUMB_CMC], kp[THUMB_IP] - kp[THUMB_MCP])
                      + _angle(kp[THUMB_IP] - kp[THUMB_MCP], kp[THUMB_TIP] - kp[THUMB_IP]))
        d = kp[THUMB_MCP] - kp[THUMB_CMC]
        nd = np.linalg.norm(d)
        thumb_rot = 0.0
        if nd > 1e-9:
            thumb_rot = float(np.arcsin(
                np.clip(np.dot(d / nd, palm_normal(kp, bool(obs.is_right))), -1.0, 1.0)))
        return np.array(curls + [thumb_bend, thumb_rot], dtype=float)

    def targets(self, obs, t: float, raw: Optional[np.ndarray] = None) -> np.ndarray:
        """Smoothed closed-ratios for the 6 DOF, ready for InspireHand.apply().

        ``raw`` reuses angles the caller already computed via raw() (for diagnostics).
        """
        ratio = self.calib.ratio(self.raw(obs) if raw is None else raw)
        if self._filt is None:
            self._filt = OneEuroFilter(self.min_cutoff, self.beta)
        return np.clip(self._filt(ratio, t), 0.0, 1.0)
