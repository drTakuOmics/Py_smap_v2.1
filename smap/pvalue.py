"""Anisotropic-Gaussian p-value metric for 2DTM peaks (new; not part of v2.1).

Zhang K. et al., IUCrJ 12 (2025): the per-pixel maximum CC over orientations (MIP) and its
pixelwise z-score (scaled MIP = SMAP's flat-fielded score) are each rank-probit
transformed over all valid pixels; the joint background is modelled as a zero-mean
anisotropic Gaussian from the second moments, and a peak's p-value is the analytic tail
probability restricted to the first quadrant. Same computation as Leopard-EM's
``pvalue_metric.py`` (commit 7819067), rewritten in NumPy; -ln p is evaluated in log form
so strong peaks do not underflow to inf.

Derived from Leopard-EM, BSD 3-Clause License,
Copyright (c) 2024-2025, Josh Dickerson and Matthew Giammar (see THIRD_PARTY_NOTICES.md).

Output is -ln(p) (natural log); Leopard-EM and the 2DTM postprocess tool use cutoffs ~8.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtri


def probit(x: np.ndarray) -> np.ndarray:
    """Rank-based probit: Phi^-1((rank - 0.5) / n), ranks 1..n (ties by order)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    ranks = np.empty(n)
    ranks[np.argsort(x, kind="stable")] = np.arange(1, n + 1)
    return ndtri((ranks - 0.5) / max(1, n))


def neg_log_p(mip: np.ndarray, scaled_mip: np.ndarray, quadrant: int = 1) -> np.ndarray:
    """-ln p for paired arrays (any shape); see module docstring."""
    shape = np.shape(mip)
    x1 = probit(scaled_mip)
    x2 = probit(mip)
    m = np.stack([x1, x2])
    d, u = np.linalg.eig(m @ m.T / x1.size)
    d, u = d.real, u.real.copy()
    a, b = math.sqrt(d.max()), math.sqrt(d.min())
    if d[0] <= d[1]:
        u = u[:, [1, 0]]
    if u[1, 0] < 0 and u[0, 0] < 0:
        u[:, 0] *= -1
    w = math.atan2(u[1, 0], u[0, 0])
    gamma = math.atan2(1.0, 0.5 * math.sin(2 * w) * (b / a - a / b))
    y0 = (math.cos(w) * x1 + math.sin(w) * x2) / max(a, 1e-12)
    y1 = (-math.sin(w) * x1 + math.cos(w) * x2) / max(b, 1e-12)
    if quadrant == 1:
        mask = (x1 > 0) & (x2 > 0)
    elif quadrant == 3:
        mask = (x1 > 0) | (x2 > 0)
    else:
        raise ValueError("quadrant must be 1 or 3")
    out = np.zeros(x1.size)
    # -ln(exp(-r2/2) * gamma / 2pi), written in log form to avoid underflow for strong peaks
    out[mask] = 0.5 * (y0[mask] ** 2 + y1[mask] ** 2) - math.log(gamma / (2 * math.pi))
    return out.reshape(shape)


def pvalue_map(mip: np.ndarray, scaled: np.ndarray, valid: np.ndarray, quadrant: int = 1) -> np.ndarray:
    """-ln p over the ``valid`` pixels of 2-D maps (0 elsewhere)."""
    out = np.zeros(mip.shape)
    out[valid] = neg_log_p(mip[valid], scaled[valid], quadrant)
    return out
