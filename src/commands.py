"""Command handlers behind the thin CLI in scripts/teleop.py. Handlers stay small."""
from __future__ import annotations

import time
from pathlib import Path

import cv2

from . import paths
from .log import get_logger

logger = get_logger(__name__)


def _resolve_source(arg_source):
    """Return the user --source, or the bundled dev sample if present, else None (caller errors).

    Portable: on machines without the bundled sample, a --source is required (e.g. 'realsense'
    for a D435, or '0' for a webcam).
    """
    if arg_source is not None:
        return str(arg_source)
    if paths.SAMPLE_VIDEO.exists():
        return str(paths.SAMPLE_VIDEO)
    return None


def _source_stem(source: str) -> str:
    return f"cam{source}" if str(source).isdigit() else Path(str(source)).stem


def cmd_wilor_image(args) -> int:
    """Phase 0 validation: run WiLoR on a still image and save an annotated overlay."""
    from .perception import WiLoREstimator
    from .perception.vis import draw_hands

    image_path = Path(args.image) if args.image else (paths.SAMPLE_IMAGES_DIR / "test1.jpg")
    if not image_path.exists():
        logger.error("image not found: %s", image_path)
        return 1

    image = cv2.imread(str(image_path))
    if image is None:
        logger.error("failed to read image: %s", image_path)
        return 1
    logger.info("input: %s (%dx%d)", image_path, image.shape[1], image.shape[0])

    est = WiLoREstimator(device=args.device, dtype=args.dtype)

    # warm-up + timed run
    est.predict(image)
    t0 = time.perf_counter()
    hands = est.predict(image)
    dt = (time.perf_counter() - t0) * 1000.0

    logger.info("detected %d hand(s) in %.1f ms", len(hands), dt)
    for i, h in enumerate(hands):
        wp = h.wrist_pos_cam
        logger.info(
            "  hand[%d] %s  wrist=(%.3f,%.3f,%.3f)m  pinch=%.1fmm  focal=%.1f",
            i, "R" if h.is_right else "L", wp[0], wp[1], wp[2],
            h.pinch_dist * 1000.0, h.focal_length,
        )

    paths.ensure_workspace()
    out_path = Path(args.out) if args.out else (paths.OUTPUT_DIR / f"{image_path.stem}_wilor.jpg")
    cv2.imwrite(str(out_path), draw_hands(image, hands))
    logger.info("overlay saved: %s", out_path)
    return 0


def cmd_live(args) -> int:
    """Phase 1: real-time hand tracking over a webcam / video, with One-Euro smoothing."""
    from .perception.realtime import run_live

    paths.ensure_workspace()
    source = _resolve_source(args.source)
    if source is None:
        logger.error("no --source given and no bundled sample found; pass --source realsense|0|<video>")
        return 1
    record = args.record
    if record is None and not args.display:
        # Headless default: produce a visual artifact so the run is verifiable.
        record = str(paths.OUTPUT_DIR / f"{_source_stem(source)}_live.mp4")

    run_live(
        source=source,
        record=record,
        display=args.display,
        max_frames=args.max_frames,
        min_cutoff=args.min_cutoff,
        beta=args.beta,
        primary=args.primary,
        device=args.device,
        dtype=args.dtype,
        proc_max_side=(args.proc_max_side or None),  # 0 => disable downscaling
    )
    return 0


def cmd_sim(args) -> int:
    """Phase 2: drive the MuJoCo xArm7 from hand teleop (retargeting validation, no hardware)."""
    from .sim.run import run_sim

    paths.ensure_workspace()
    source = _resolve_source(args.source)
    if source is None:
        logger.error("no --source given and no bundled sample found; pass --source realsense|0|<video>")
        return 1
    record = args.record
    if record is None:
        record = str(paths.OUTPUT_DIR / f"{_source_stem(source)}_sim.mp4")

    run_sim(
        source=source,
        record=record,
        display=args.display,
        max_frames=args.max_frames,
        scale=args.scale,
        depth_scale=args.depth_scale,
        pos_only=args.pos_only,
        primary=args.primary,
        min_cutoff=args.min_cutoff,
        beta=args.beta,
        proc_max_side=(args.proc_max_side or None),
        device=args.device,
        dtype=args.dtype,
    )
    return 0


def cmd_teleop(args) -> int:
    """Phase 3: drive the real xArm7 with the shared teleop loop (dry-run unless --execute)."""
    from .control.safety import SafetyLimiter
    from .control.xarm_controller import XArm7Controller
    from .loop import run_teleop
    from .retarget import Retargeter

    dry_run = not args.execute
    if not dry_run and not args.ip:
        logger.error("--execute requires --ip <xArm controller IP>")
        return 1
    if not dry_run:
        logger.warning("EXECUTE mode: commanding a REAL robot. Ensure E-stop is within reach.")

    paths.ensure_workspace()
    source = _resolve_source(args.source)
    if source is None:
        logger.error("no --source given and no bundled sample found; pass --source realsense|0|<video>")
        return 1
    record = args.record
    if record is None:
        record = str(paths.OUTPUT_DIR / f"{_source_stem(source)}_teleop.mp4")

    backend = XArm7Controller(ip=args.ip or "0.0.0.0", dry_run=dry_run, tcp_speed=args.tcp_speed)
    retarget = Retargeter(scale=args.scale, depth_scale=args.depth_scale, pos_only=args.pos_only)
    safety = SafetyLimiter(max_step_m=args.max_step_m)
    run_teleop(
        backend=backend, source=source, retarget=retarget, safety=safety, record=record,
        display=args.display, max_frames=args.max_frames, primary=args.primary,
        min_cutoff=args.min_cutoff,
        beta=args.beta, proc_max_side=(args.proc_max_side or None),
        device=args.device, dtype=args.dtype,
    )
    return 0
