"""One-Euro filter (Casiez et al. 2012) for low-latency jitter suppression.

Works on scalars or fixed-length vectors (numpy). The adaptive cutoff trades a little lag at
high speed for strong smoothing when nearly still -- the right behavior for hand teleop.
"""
from __future__ import annotations

import math

import numpy as np


class _LowPass:
    def __init__(self) -> None:
        self.y: np.ndarray | None = None

    def __call__(self, x: np.ndarray, alpha: float) -> np.ndarray:
        if self.y is None:
            self.y = np.array(x, dtype=float)
        else:
            self.y = alpha * x + (1.0 - alpha) * self.y
        return self.y


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = _LowPass()
        self._dx = _LowPass()
        self._t_prev: float | None = None
        self._x_prev: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = x
            self._x.y = x.copy()
            return x
        dt = t - self._t_prev
        if dt <= 0:
            dt = 1e-3
        dx = (x - self._x_prev) / dt
        edx = self._dx(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(edx))
        ex = self._x(x, self._alpha(cutoff, dt))
        self._t_prev = t
        self._x_prev = x
        return ex
