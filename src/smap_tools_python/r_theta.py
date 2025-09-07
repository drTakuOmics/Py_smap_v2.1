"""Compute an image's mean intensity in ``(r, \theta)`` bins."""
from __future__ import annotations

import numpy as np

from .nm import nm


def r_theta(image: np.ndarray, n_theta: int = 360) -> np.ndarray:
    """Bin ``image`` into radius/angle coordinates.

    This function mirrors SMAP's MATLAB ``rTheta`` helper but performs all
    operations with vectorized NumPy calls rather than explicit loops.

    Parameters
    ----------
    image : ndarray
        Input square image.
    n_theta : int, optional
        Number of angular bins. Default is 360.

    Returns
    -------
    out : ndarray
        Array of shape ``(n_theta, n_r)`` containing the mean intensity of
        ``image`` within each ``(r, \theta)`` bin. ``n_r`` depends on the
        image size.
    """

    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("image must be square")
    N = image.shape[0]

    y, x = np.indices(image.shape)
    cx = cy = N // 2
    x = x - cx
    y = cy - y  # MATLAB uses a flipped Y axis

    r_coord = np.hypot(x, y)
    t_coord = np.mod(np.arctan2(y, x), 2 * np.pi)

    if N % 2:
        r_bins = np.linspace(0, np.sqrt(2) / 2, int(N * np.sqrt(2) / 2) + 1)
    else:
        r_bins = np.linspace(0, np.sqrt(2) / 2, int((N + 1) * np.sqrt(2) / 2) + 1)[:-1]
    r_bins *= N
    t_bins = np.linspace(0, 2 * np.pi, n_theta + 1)

    sums, _, _ = np.histogram2d(t_coord.ravel(), r_coord.ravel(), bins=[t_bins, r_bins], weights=image.ravel())
    counts, _, _ = np.histogram2d(t_coord.ravel(), r_coord.ravel(), bins=[t_bins, r_bins])
    with np.errstate(invalid="ignore"):
        out = sums / counts
    mean_val = np.nanmean(out)
    out = np.where(np.isnan(out), mean_val, out)
    return nm(out)
