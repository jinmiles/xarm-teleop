"""Shared logger for status/progress output. Use this instead of raw print for new status."""
from __future__ import annotations

import logging

_CONFIGURED = False


def get_logger(name: str = "teleop") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
