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
