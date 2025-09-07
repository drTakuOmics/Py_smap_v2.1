import numpy as np
from scipy.spatial.transform import Rotation as R
from .rotate import normalize_rotation_matrices


def frealign2smap(ori):
    """Convert FREALIGN Euler triplets to rotation matrices and quaternions."""
    ang = np.asarray(ori, dtype=float)
    if ang.ndim != 2 or ang.shape[1] != 3:
        raise ValueError("ori must be of shape (N,3)")
    base = R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
    rot = R.from_euler("ZYZ", np.deg2rad(ang[:, ::-1])).as_matrix()
    rot = base @ rot
    rot = normalize_rotation_matrices(rot)
    q = R.from_matrix(rot).as_quat()  # x, y, z, w
    q_out = np.vstack((q[:, 3], q[:, 0], q[:, 1], q[:, 2]))
    return rot.transpose(1, 2, 0), q_out
