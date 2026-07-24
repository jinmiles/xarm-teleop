"""Command handlers behind the thin CLI in scripts/teleop.py. Handlers stay small."""
from __future__ import annotations

import time
from pathlib import Path

import cv2

from . import paths
from .log import get_logger

logger = get_logger(__name__)


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
    source = args.source if args.source is not None else str(paths.EXTERNAL_ROOT / "HaWoR" / "example" / "video_0.mp4")
    record = args.record
    if record is None and not args.display:
        # Headless default: produce a visual artifact so the run is verifiable.
        stem = Path(str(source)).stem if not str(source).isdigit() else f"cam{source}"
        record = str(paths.OUTPUT_DIR / f"{stem}_live.mp4")

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
    source = args.source if args.source is not None else str(
        paths.EXTERNAL_ROOT / "HaWoR" / "example" / "video_0.mp4")
    record = args.record
    if record is None:
        stem = Path(str(source)).stem if not str(source).isdigit() else f"cam{source}"
        record = str(paths.OUTPUT_DIR / f"{stem}_sim.mp4")

    run_sim(
        source=source,
        record=record,
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
