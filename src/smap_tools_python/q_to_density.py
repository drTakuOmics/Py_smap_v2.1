"""Estimate rotation density via 3-D histograms."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def q_to_density(q_a, q_b=None, n_bins=64):
    """Compute density volumes for sets of rotation matrices.

    Parameters
    ----------
    q_a : array_like, shape (..., 3, 3)
        Rotation matrices of interest.
    q_b : array_like, shape (..., 3, 3), optional
        Optional control set of rotation matrices. If provided, a second
        density volume is returned for comparison.
    n_bins : int, optional
        Number of bins per axis for the 3-D histogram. Defaults to 64.

    Returns
    -------
    V_a : ndarray, shape (n_bins, n_bins, n_bins)
        Density of ``q_a`` in rotation-vector space.
    V_b : ndarray or None
        Density of ``q_b`` if supplied, otherwise ``None``.
    """

    def _prep(q):
        q = np.asarray(q, dtype=float)
        if q.shape[-2:] != (3, 3):
            raise ValueError("rotation matrices must have shape (..., 3, 3)")
        return q.reshape(-1, 3, 3)

    q_a = _prep(q_a)
    rv_a = R.from_matrix(q_a).as_rotvec()
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    V_a, edges = np.histogramdd(rv_a, bins=(edges, edges, edges))

    V_b = None
    if q_b is not None:
        q_b = _prep(q_b)
        rv_b = R.from_matrix(q_b).as_rotvec()
        V_b, _ = np.histogramdd(rv_b, bins=(edges[0], edges[1], edges[2]))

    return V_a.astype(float), None if V_b is None else V_b.astype(float)


__all__ = ["q_to_density"]
