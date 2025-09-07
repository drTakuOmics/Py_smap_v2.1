import numpy as np
from scipy.spatial.transform import Rotation as R

from .rotate import normalize_rotation_matrices


def smap2cistem(rot_mats):
    """Convert SMAP rotation matrices to cisTEM Euler triplets.

    Parameters
    ----------
    rot_mats : ndarray, shape (..., 3, 3)
        Rotation matrices to convert.

    Returns
    -------
    ndarray, shape (N, 3)
        cisTEM-style Euler angles ``(phi, theta, psi)`` in degrees.
    """

    mats = np.asarray(rot_mats, dtype=float)
    if mats.ndim == 2:
        mats = mats[None, ...]
    if mats.shape[0] == 3 and mats.shape[1] == 3:
        mats = mats.transpose(2, 0, 1)
    mats = normalize_rotation_matrices(mats)
    mats = np.transpose(mats, (0, 2, 1))  # transpose each matrix

    angles = R.from_matrix(mats).as_euler('ZYZ')
    return np.rad2deg(angles[:, ::-1])

