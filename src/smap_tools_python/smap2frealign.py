import numpy as np
from scipy.spatial.transform import Rotation as R

from .rotate import normalize_rotation_matrices


def smap2frealign(rot_mats):
    """Convert SMAP rotation matrices to FREALIGN Euler triplets.

    Parameters
    ----------
    rot_mats : ndarray, shape (..., 3, 3)
        Rotation matrices to convert.

    Returns
    -------
    ori : ndarray, shape (N, 3)
        FREALIGN-style Euler angles ``(phi, theta, psi)`` in degrees.
    q_out : ndarray, shape (4, N)
        Corresponding quaternions in ``(w, x, y, z)`` order.
    """

    mats = np.asarray(rot_mats, dtype=float)
    if mats.ndim == 2:
        mats = mats[None, ...]
    if mats.shape[0] == 3 and mats.shape[1] == 3:
        mats = mats.transpose(2, 0, 1)
    mats = normalize_rotation_matrices(mats)

    rot = R.from_matrix(mats)
    qr = R.from_euler('xyz', [180, 0, 0], degrees=True)
    q_out_rot = qr * rot
    angles = q_out_rot.as_euler('ZYZ')
    ori = np.rad2deg(angles[:, ::-1])

    q_out = q_out_rot.as_quat()  # x, y, z, w
    q_out = np.vstack((q_out[:, 3], q_out[:, 0], q_out[:, 1], q_out[:, 2]))
    return ori, q_out

