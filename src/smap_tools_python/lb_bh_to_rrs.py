"""Convert Lab/Beam and Body/Head rotations to RRS coordinates."""

import numpy as np
from scipy.spatial.transform import Rotation as R


def lb_bh_to_rrs(q_lb, q_bh, coeff_b, mu_b, coeff_h, mu_h):
    """Map rotation matrices to reduced rotation space (RRS).

    Parameters
    ----------
    q_lb : array_like, shape (N, 3, 3) or (3, 3, N)
        Rotation matrices in the lab-to-beam frame.
    q_bh : array_like, shape (N, 3, 3) or (3, 3, N)
        Rotation matrices in the body-to-head frame.
    coeff_b, coeff_h : array_like, shape (3, 3)
        PCA coefficient matrices for the two frames.
    mu_b, mu_h : array_like, shape (3,)
        Mean rotation vectors for the two frames.

    Returns
    -------
    ndarray, shape (N, 3)
        Coordinates in reduced rotation space in degrees.
    """

    q_lb = np.asarray(q_lb, dtype=float)
    q_bh = np.asarray(q_bh, dtype=float)
    coeff_b = np.asarray(coeff_b, dtype=float)
    coeff_h = np.asarray(coeff_h, dtype=float)
    mu_b = np.asarray(mu_b, dtype=float)
    mu_h = np.asarray(mu_h, dtype=float)

    if q_lb.shape[-2:] != (3, 3) or q_bh.shape[-2:] != (3, 3):
        raise ValueError("Input rotations must have shape (..., 3, 3)")

    # Move rotation index to first dimension if necessary
    q_lb = np.reshape(q_lb, (-1, 3, 3)) if q_lb.shape[0] != 3 else np.moveaxis(q_lb, -1, 0)
    q_bh = np.reshape(q_bh, (-1, 3, 3)) if q_bh.shape[0] != 3 else np.moveaxis(q_bh, -1, 0)

    zero_lb = (-mu_b) @ coeff_b
    zero_bh = (-mu_h) @ coeff_h

    n = q_lb.shape[0]
    rrs = np.empty((n, 3), dtype=float)

    for i in range(n):
        rv_lb = R.from_matrix(q_lb[i]).as_rotvec()
        coords_lb = (rv_lb - mu_b) @ coeff_b - zero_lb

        rv_bh = R.from_matrix(q_bh[i]).as_rotvec()
        coords_bh = (rv_bh - mu_h) @ coeff_h - zero_bh

        rrs[i] = [coords_lb[0], coords_lb[1], coords_bh[0]]

    return rrs * (-180.0 / np.pi)
