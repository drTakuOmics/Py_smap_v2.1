import numpy as np
from scipy.spatial.transform import Rotation as R
from .measure_qd import measure_qd


def gridded_qs(angle_range: float, inc: float) -> np.ndarray:
    """Generate a grid of rotation matrices within ``angle_range`` degrees.

    Parameters
    ----------
    angle_range : float
        Maximum angular deviation from the identity in degrees.
    inc : float
        Step size in degrees for the underlying rotation-vector grid.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, 3, 3)`` containing rotation matrices whose
        quaternion distance from the identity does not exceed ``angle_range``.
    """
    base = np.deg2rad(np.arange(-angle_range, angle_range + 1e-9, inc))
    base -= base.mean()
    xd, yd, zd = np.meshgrid(base, base, base, indexing="ij")
    rotvecs = np.stack([xd.ravel(), yd.ravel(), zd.ravel()], axis=1)
    rot = R.from_rotvec(rotvecs)
    mats = rot.as_matrix()
    quats = rot.as_quat()
    quats = np.concatenate([quats[:, 3:4], quats[:, :3]], axis=1)  # w, x, y, z
    qd = measure_qd([1.0, 0.0, 0.0, 0.0], quats)
    return mats[qd <= angle_range]
