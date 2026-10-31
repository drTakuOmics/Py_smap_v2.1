"""Joint (MIP, z-score) p-value for 2DTM peaks (new; not part of v2.1).

Method of Zhang K. et al., IUCrJ 12 (2025), implemented here from its mathematical
definition:

1. Over all valid pixels, the per-pixel maximum CC over orientations (MIP) and its
   pixelwise z-score (SMAP's flat-fielded score) are each mapped to standard-normal
   quantiles by a rank (probit) transform: g = Phi^-1((rank - 1/2) / n).
2. The background of the pair x = (g_z, g_mip) is modelled as a zero-mean bivariate
   Gaussian with second-moment matrix C = <x x^T>.
3. A pixel's p-value is the background probability of being at least as extreme in both
   coordinates: the Gaussian mass beyond its Mahalanobis radius r (r^2 = x^T C^-1 x) that
   lies inside the positive quadrant. Whitening (x -> C^-1/2 x) maps that quadrant onto a
   sector of opening angle gamma, and in whitened coordinates the radius is chi-distributed
   with 2 degrees of freedom and the angle uniform, so

       p = exp(-r^2 / 2) * gamma / (2 pi),     gamma = angle between C^-1/2 e1 and C^-1/2 e2.

Pixels outside the positive quadrant get p = 1. The output is -ln p (cutoffs of ~8 are used
in the literature). Written from the equations above; for the default (positive quadrant) it
reproduces the published reference implementation (Leopard-EM) to floating-point precision.
quadrant=3 (either coordinate positive) uses the sector angle 2*pi - gamma.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import sqrtm
from scipy.special import ndtri


def probit(x: np.ndarray) -> np.ndarray:
    """Rank-based inverse-normal transform: Phi^-1((rank - 0.5) / n), ranks 1..n (ties by order)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    ranks = np.empty(n)
    ranks[np.argsort(x, kind="stable")] = np.arange(1, n + 1)
    return ndtri((ranks - 0.5) / max(1, n))


def neg_log_p(mip: np.ndarray, scaled_mip: np.ndarray, quadrant: int = 1) -> np.ndarray:
    """-ln p for paired arrays (any shape); quadrant 1 = both coordinates positive,
    quadrant 3 = either coordinate positive (three quadrants)."""
    shape = np.shape(mip)
    X = np.stack([probit(scaled_mip), probit(mip)], axis=1)          # (n, 2)
    C = X.T @ X / len(X)                                             # second moments
    W = np.real(np.linalg.inv(sqrtm(C)))                             # whitening, symmetric
    r2 = np.einsum("ni,ij,nj->n", X, np.linalg.inv(C), X)
    a, b = W[:, 0], W[:, 1]                                          # images of e1, e2
    open_angle = math.acos(float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))
    if quadrant == 1:
        gamma, inside = open_angle, (X[:, 0] > 0) & (X[:, 1] > 0)
    elif quadrant == 3:
        # complement of the negative quadrant, whose image has the same opening angle
        gamma, inside = 2 * math.pi - open_angle, (X[:, 0] > 0) | (X[:, 1] > 0)
    else:
        raise ValueError("quadrant must be 1 or 3")
    out = np.zeros(len(X))
    out[inside] = 0.5 * r2[inside] - math.log(gamma / (2 * math.pi))
    return out.reshape(shape)


def pvalue_map(mip: np.ndarray, scaled: np.ndarray, valid: np.ndarray, quadrant: int = 1) -> np.ndarray:
    """-ln p over the ``valid`` pixels of 2-D maps (0 elsewhere)."""
    out = np.zeros(mip.shape)
    out[valid] = neg_log_p(mip[valid], scaled[valid], quadrant)
    return out
