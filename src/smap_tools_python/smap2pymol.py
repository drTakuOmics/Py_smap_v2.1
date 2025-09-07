import numpy as np
from scipy.spatial.transform import Rotation as R

from .rotate import normalize_rotation_matrices


def smap2pymol(rot_mats):
    """Convert rotation matrices to PyMOL axis-angle notation.

    Parameters
    ----------
    rot_mats : ndarray, shape (..., 3, 3)
        Rotation matrices in SMAP convention.

    Returns
    -------
    ndarray, shape (N, 4)
        Array where each row contains ``(ax, ay, az, angle_deg)``.
    """

    mats = np.asarray(rot_mats, dtype=float)
    if mats.ndim == 2:
        mats = mats[None, ...]
    if mats.shape[0] == 3 and mats.shape[1] == 3:
        mats = mats.transpose(2, 0, 1)
    mats = normalize_rotation_matrices(mats)

    rot = R.from_matrix(mats)
    rotvecs = rot.as_rotvec()  # axis * angle in radians
    angles = np.linalg.norm(rotvecs, axis=1)
    axes = np.divide(
        rotvecs,
        angles[:, None],
        out=np.zeros_like(rotvecs),
        where=angles[:, None] != 0,
    )
    return np.column_stack((axes, np.degrees(angles)))

