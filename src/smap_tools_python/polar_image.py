"""Convert an image to log-polar coordinates.

This module provides a translation of SMAP's ``polarImage`` MATLAB
utility. It maps a 2-D Cartesian image into a log-radius / angle grid
using spline interpolation from :mod:`scipy.ndimage`.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def polar_image(
    image: np.ndarray,
    n_rho: int,
    n_theta: int,
    method: str = "linear",
    center: tuple[float, float] | None = None,
    shape: str = "full",
) -> tuple[np.ndarray, np.ndarray]:
    """Warp ``image`` into log-polar coordinates.

    Parameters
    ----------
    image : ndarray
        Input 2-D array.
    n_rho : int
        Number of radial samples.
    n_theta : int
        Number of angular samples.
    method : {"nearest", "linear", "bicubic"}, optional
        Interpolation method. Defaults to ``"linear"``.
    center : tuple of float, optional
        ``(x, y)`` coordinates of the origin. Defaults to the image centre.
    shape : {"full", "valid"}, optional
        Behaviour for determining the maximum radius. ``"full"`` uses the
        distance to the farthest corner whereas ``"valid"`` uses the distance
        to the nearest edge. Defaults to ``"full"``.

    Returns
    -------
    polar : ndarray
        Polar representation with shape ``(n_rho, n_theta)``.
    rho : ndarray
        Radial coordinates corresponding to the rows of ``polar``.
    """

    image = np.asarray(image, dtype=float)
    ar, ac = image.shape
    if center is None:
        center = ((ac - 1) / 2.0, (ar - 1) / 2.0)
    cx, cy = center

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    if shape == "full":
        corners = np.array([[0, 0], [ac - 1, 0], [ac - 1, ar - 1], [0, ar - 1]], dtype=float)
        d = np.max(np.sqrt(((corners - (cx, cy)) ** 2).sum(axis=1)))
    elif shape == "valid":
        d = min(cx, ac - 1 - cx, cy, ar - 1 - cy)
    else:  # pragma: no cover - defensive programming
        raise ValueError("shape must be 'full' or 'valid'")

    rho = np.logspace(np.log10(1.0), np.log10(d), n_rho)

    rr = rho[:, None]
    tt = theta[None, :]
    xx = rr * np.cos(tt) + cx
    yy = rr * np.sin(tt) + cy

    orders = {"nearest": 0, "linear": 1, "bicubic": 3}
    order = orders.get(method.lower())
    if order is None:  # pragma: no cover - handled by tests
        raise ValueError(f"Unknown interpolation method: {method}")

    coords = np.vstack([yy.ravel(), xx.ravel()])
    polar = map_coordinates(
        image,
        coords,
        order=order,
        mode="constant",
        cval=0.0,
    ).reshape(n_rho, n_theta)
    return polar, rho
