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

    dex_hand, dex_retarget = _build_dex_hand(args)
    run_sim(
        source=source,
        record=record,
        dex_hand=dex_hand,
        dex_retarget=dex_retarget,
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

    # A dexterous hand occupies the tool flange, so the UFACTORY gripper is not there to talk to:
    # enabling it would latch controller error 19 and freeze every subsequent servo command.
    use_gripper = not (args.no_gripper or args.hand_port)
    if not use_gripper and not args.no_gripper:
        logger.info("--hand-port given: skipping the UFACTORY gripper (pass --gripper to override)")
    backend = XArm7Controller(ip=args.ip or "0.0.0.0", dry_run=dry_run, tcp_speed=args.tcp_speed,
                              use_gripper=use_gripper)
    retarget = Retargeter(scale=args.scale, depth_scale=args.depth_scale, pos_only=args.pos_only)
    safety = SafetyLimiter(max_step_m=args.max_step_m)
    dex_hand, dex_retarget = _build_dex_hand(args)
    run_teleop(
        backend=backend, source=source, retarget=retarget, safety=safety,
        dex_hand=dex_hand, dex_retarget=dex_retarget, record=record,
        display=args.display, max_frames=args.max_frames, primary=args.primary,
        min_cutoff=args.min_cutoff,
        beta=args.beta, proc_max_side=(args.proc_max_side or None),
        device=args.device, dtype=args.dtype,
    )
    return 0


def _build_dex_hand(args):
    """Build the Inspire hand driver + finger retargeter, or (None, None) if not requested."""
    from .control.inspire_hand import InspireHand
    from .retarget import DexCalibration, DexHandRetargeter

    port = getattr(args, "hand_port", None)
    if not port and not getattr(args, "hand_dry_run", False):
        return None, None
    calib_path = Path(args.hand_calib) if args.hand_calib else paths.HAND_CALIB
    if calib_path.exists():
        from .retarget.dex_hand import DOF_NAMES as _DEX_NAMES

        calib = DexCalibration.load(calib_path)
        # A narrow span saturates ratio() to a constant, which reaches the hand as one frame the
        # deadband then suppresses -- i.e. fingers that never move, with nothing else to show why.
        span = calib.closed_rad - calib.open_rad
        logger.info("dex hand calibration: %s (open->fist span %s rad)",
                    calib_path, ", ".join(f"{n}={s:+.2f}" for n, s in zip(_DEX_NAMES, span)))
        narrow = [n for n, s in zip(_DEX_NAMES, span) if abs(s) < 0.15]
        if narrow:
            logger.warning("calibration span is tiny for %s: those DOF will barely move. "
                           "Re-run 'teleop.py hand-calib' with a wider open/fist difference",
                           ", ".join(narrow))
    else:
        calib = None
        logger.warning("no hand calibration at %s; using rough defaults "
                       "(run 'teleop.py hand-calib' for your hand)", calib_path)
    hand = InspireHand(port=port or "/dev/ttyUSB0", baud=args.hand_baud, hand_id=args.hand_id,
                       speed=args.hand_speed, force=args.hand_force,
                       speed_reg=args.hand_speed_reg, force_reg=args.hand_force_reg,
                       dry_run=getattr(args, "hand_dry_run", False))
    if not hand.dry_run:
        logger.warning("dex hand LIVE on %s: fingers will move with your hand", hand.port)
    return hand, DexHandRetargeter(calib=calib)


def cmd_hand_test(args) -> int:
    """Bring-up check for the Inspire hand: read state, then sweep each DOF open->closed->open."""
    import time as _time

    from .control.inspire_hand import DOF_NAMES, InspireHand

    hand = InspireHand(port=args.port, baud=args.baud, hand_id=args.id, speed=args.speed,
                       force=args.force, speed_reg=args.speed_reg, force_reg=args.force_reg,
                       dry_run=args.dry_run)
    try:
        hand.connect()
        logger.info("opening hand ...")
        hand.open_hand()
        _time.sleep(args.hold)
        for i, name in enumerate(DOF_NAMES):
            logger.info("[%d/%d] bending %s (others stay open)", i + 1, len(DOF_NAMES), name)
            angles = hand.open_cmd.copy()
            angles[i] = hand.closed_cmd[i]
            hand.set_angles(angles)
            _time.sleep(args.hold)
            back = hand.read_angle_set()
            if back is not None:
                logger.info("      setpoint readback %s",
                            ", ".join(f"{n}={v}" for n, v in zip(DOF_NAMES, back)))
            hand.open_hand()
            _time.sleep(args.hold)
        logger.info("sweep done. Confirm each named DOF moved the finger it claims, and that the "
                    "bent pose really is a light grip (open %s -> closed %s in device units).",
                    hand.open_cmd.tolist(), hand.closed_cmd.tolist())
    finally:
        hand.close()
    return 0


def cmd_hand_calib(args) -> int:
    """Record this operator's open-hand and fist finger angles into a calibration JSON."""
    import time as _time

    import numpy as np

    from .camera import open_source
    from .perception.realtime import HandTracker
    from .perception.wilor_estimator import WiLoREstimator
    from .retarget import DexCalibration, DexHandRetargeter
    from .retarget.dex_hand import DOF_NAMES

    paths.ensure_workspace()
    source = _resolve_source(args.source)
    if source is None:
        logger.error("no --source given and no bundled sample found; pass --source realsense|0|<video>")
        return 1
    out_path = Path(args.out) if args.out else paths.HAND_CALIB

    est = WiLoREstimator(device=args.device, dtype=args.dtype, proc_max_side=args.proc_max_side or None,
                         primary=args.primary)
    tracker = HandTracker(est)
    retarget = DexHandRetargeter()
    poses = {"open": "hold your hand FULLY OPEN, fingers straight, thumb out",
             "closed": "make a TIGHT FIST with the thumb across the palm"}
    captured: dict[str, np.ndarray] = {}

    with open_source(source) as cam:
        frames = cam.frames()
        for key, prompt in poses.items():
            logger.info("=== %s pose: %s ===", key.upper(), prompt)
            for sec in range(args.countdown, 0, -1):
                logger.info("  capturing in %d ...", sec)
                t_end = _time.perf_counter() + 1.0
                while _time.perf_counter() < t_end:
                    next(frames, None)
            samples = []
            while len(samples) < args.frames:
                frame = next(frames, None)
                if frame is None:
                    logger.error("camera ran out of frames during calibration")
                    return 1
                _, prim = tracker.update(frame.color, frame.timestamp)
                if prim is not None:
                    samples.append(retarget.raw(prim))
            captured[key] = np.median(np.stack(samples), axis=0)
            logger.info("  %s: %s", key,
                        ", ".join(f"{n}={v:+.2f}" for n, v in zip(DOF_NAMES, captured[key])))

    calib = DexCalibration(captured["open"], captured["closed"])
    span = calib.closed_rad - calib.open_rad
    for name, s in zip(DOF_NAMES, span):
        if abs(s) < 0.15:
            logger.warning("DOF %s has a tiny open/closed span (%.2f rad) - recapture it", name, s)
    calib.save(out_path)
    logger.info("calibration saved: %s", out_path)
    return 0
