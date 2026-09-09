"""Noitom glove skeleton -> 21 MANO-ordered keypoints (wrist frame, meters).

The glove is an IMU device: it streams per-bone *rotations* only, never a position in space
(the wrist's absolute position is what the depth camera provides in method 2). Forward
kinematics over the streamed bone rotations therefore yields a hand shape, not a hand location.

Emitting MANO's 21-keypoint layout is deliberate: every downstream stage (finger retargeting,
calibration, the overlay) already speaks that layout, so the glove drops in where WiLoR sits
without a second retargeting path.

Bone names follow the Noitom / Axis Studio convention used by the glove stream:
    <side>Hand                  wrist (chain root)
    <side>HandThumb1..3         thumb joints
    <side>InHand<finger>        metacarpal inside the palm (finger spread)
    <side>Hand<finger>1..3      proximal / middle / distal phalanx
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

SIDE_PREFIX = {"right": "Right", "left": "Left"}
FINGERS = ("Index", "Middle", "Ring", "Pinky")

# The distal phalanx has no child bone in the stream, so the fingertip is extrapolated from the
# distal joint along its own bone direction, at this fraction of the incoming bone's length.
TIP_RATIO = 0.75

# A hand whose middle-finger chain is longer than this cannot be in meters: the stream is in
# centimeters (MocapApi's other unit) and gets scaled once, at load.
_CM_THRESHOLD_M = 0.5


def wrist_bone(prefix: str) -> str:
    return f"{prefix}Hand"


def bone_parents(prefix: str) -> dict[str, str]:
    """Parent bone of every finger bone of one hand, ordered parents-before-children."""
    root = wrist_bone(prefix)
    parents: dict[str, str] = {}
    parent = root
    for name in (f"{prefix}HandThumb{i}" for i in (1, 2, 3)):
        parents[name] = parent
        parent = name
    for finger in FINGERS:
        parent = f"{prefix}InHand{finger}"
        parents[parent] = root
        for i in (1, 2, 3):
            name = f"{prefix}Hand{finger}{i}"
            parents[name] = parent
            parent = name
    return parents


def hand_bones(prefix: str) -> list[str]:
    """Every bone of one hand, wrist first, parents before children."""
    return [wrist_bone(prefix), *bone_parents(prefix)]


def mano_bones(prefix: str) -> list[str]:
    """Bone whose *origin* is each MANO keypoint; fingertips (4/8/12/16/20) are extrapolated."""
    names = [wrist_bone(prefix)] + [f"{prefix}HandThumb{i}" for i in (1, 2, 3)] + [""]
    for finger in FINGERS:
        names += [f"{prefix}Hand{finger}{i}" for i in (1, 2, 3)] + [""]
    return names


# --- fallback rest skeleton -----------------------------------------------------------------
# Only used when the stream carries no bone translations. Nominal adult right-hand geometry in
# the wrist frame (+z distal, +x radial/thumb side, +y dorsal); the left hand mirrors x.
_KNUCKLE = {"Index": (0.022, 0.078), "Middle": (0.004, 0.082),
            "Ring": (-0.016, 0.078), "Pinky": (-0.034, 0.070)}   # (spread x, palm length z)
_PHALANX = {"Index": (0.040, 0.028), "Middle": (0.045, 0.030),
            "Ring": (0.040, 0.028), "Pinky": (0.032, 0.022)}     # (proximal, middle) lengths
_THUMB = {1: (0.030, 0.0, 0.020), 2: (0.050, 0.0, 0.040), 3: (0.062, 0.0, 0.055)}


def fallback_offsets(prefix: str) -> dict[str, np.ndarray]:
    """Rest-pose parent-relative bone translations in meters, keyed by bone name."""
    mirror = -1.0 if prefix == "Left" else 1.0
    rest = {wrist_bone(prefix): np.zeros(3)}
    for finger in FINGERS:
        xk, zk = _KNUCKLE[finger]
        xk *= mirror
        prox, mid = _PHALANX[finger]
        rest[f"{prefix}InHand{finger}"] = np.array([xk * 0.5, 0.0, zk * 0.35])
        rest[f"{prefix}Hand{finger}1"] = np.array([xk, 0.0, zk])
        rest[f"{prefix}Hand{finger}2"] = np.array([xk, 0.0, zk + prox])
        rest[f"{prefix}Hand{finger}3"] = np.array([xk, 0.0, zk + prox + mid])
    for i, (x, y, z) in _THUMB.items():
        rest[f"{prefix}HandThumb{i}"] = np.array([x * mirror, y, z])
    parents = bone_parents(prefix)
    return {bone: rest[bone] - rest[parent] for bone, parent in parents.items()}


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a (w, x, y, z) quaternion, as MocapApi returns it."""
    w, x, y, z = (float(v) for v in np.asarray(q, dtype=float).reshape(4))
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
    ])


def normalize_offsets(prefix: str, offsets: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], bool]:
    """Bone translations in meters, gaps filled from the fallback skeleton.

    Returns (offsets, complete): ``complete`` is False when the stream carried no usable bone
    lengths at all, i.e. the whole hand geometry is the nominal fallback.
    """
    fallback = fallback_offsets(prefix)
    streamed = {b: np.asarray(v, dtype=float).reshape(3) for b, v in offsets.items()
                if v is not None and float(np.linalg.norm(v)) > 1e-6}
    if not streamed:
        return dict(fallback), False

    chain = [f"{prefix}InHandMiddle"] + [f"{prefix}HandMiddle{i}" for i in (1, 2, 3)]
    span = sum(float(np.linalg.norm(streamed[b])) for b in chain if b in streamed)
    if span > _CM_THRESHOLD_M:  # MocapApi's other unit is centimeters
        streamed = {b: v * 0.01 for b, v in streamed.items()}
    out = dict(fallback)
    out.update(streamed)
    return out, True


def keypoints_from_bones(
    prefix: str,
    rotations: Mapping[str, np.ndarray],
    offsets: Mapping[str, np.ndarray],
) -> Optional[np.ndarray]:
    """Forward-kinematic (21, 3) MANO keypoints in the wrist frame, meters.

    ``rotations`` maps bone -> (w, x, y, z) local rotation (parent-relative), ``offsets``
    bone -> (3,) parent-relative translation in meters. Bones absent from ``rotations`` are
    treated as unrotated, so a glove that streams fewer bones still yields a whole hand.
    """
    root = wrist_bone(prefix)
    parents = bone_parents(prefix)
    pos = {root: np.zeros(3)}
    rot = {root: np.eye(3)}
    for bone, parent in parents.items():
        if parent not in pos:
            continue
        offset = np.asarray(offsets.get(bone, np.zeros(3)), dtype=float).reshape(3)
        local = quat_to_matrix(rotations[bone]) if bone in rotations else np.eye(3)
        pos[bone] = pos[parent] + rot[parent] @ offset
        rot[bone] = rot[parent] @ local

    kp = np.zeros((21, 3), dtype=float)
    names = mano_bones(prefix)
    for i, bone in enumerate(names):
        if bone:
            if bone not in pos:
                return None
            kp[i] = pos[bone]
            continue
        # fingertip: continue past the distal joint along its own (rotated) bone direction
        distal = names[i - 1]
        offset = np.asarray(offsets.get(distal, np.zeros(3)), dtype=float).reshape(3)
        kp[i] = pos[distal] + rot[distal] @ (offset * TIP_RATIO)
    return kp
