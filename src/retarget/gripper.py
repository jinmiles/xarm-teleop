"""Map thumb-index pinch distance to a gripper closed-ratio in [0,1]."""
from __future__ import annotations

import numpy as np


def pinch_to_closed(pinch_m: float, open_at: float = 0.08, closed_at: float = 0.02) -> float:
    """Return closed ratio: 1.0 when pinched shut (<= closed_at), 0.0 when open (>= open_at)."""
    r = (open_at - pinch_m) / (open_at - closed_at)
    return float(np.clip(r, 0.0, 1.0))
