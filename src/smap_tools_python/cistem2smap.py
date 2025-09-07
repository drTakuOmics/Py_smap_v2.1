import numpy as np
from scipy.spatial.transform import Rotation as R
from .rotate import normalize_rotation_matrices


def cistem2smap(euler_angles):
    """Convert CIS TEM ``[phi, theta, psi]`` angles to SMAP rotation matrices.

    Parameters
    ----------
    euler_angles : array_like, shape (N, 3)
        Euler angles in degrees following the CIS TEM convention.  Each row is
        ``[phi, theta, psi]``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(3, 3, N)`` containing rotation matrices.
    """

    ea = np.asarray(euler_angles, dtype=float)
    if ea.ndim != 2 or ea.shape[1] != 3:
        raise ValueError("euler_angles must be of shape (N,3)")

    ang = np.deg2rad(ea[:, ::-1])
    rot = R.from_euler("ZYZ", ang).as_matrix()
    rot = rot.transpose(0, 2, 1)
    rot = normalize_rotation_matrices(rot)
    return rot.transpose(1, 2, 0)
