import numpy as np


def measure_qd(q1, q2):
    """Angular distance between quaternions in degrees.

    Parameters
    ----------
    q1 : array_like, shape (4,)
        Reference quaternion. Does not need to be normalized.
    q2 : array_like, shape (..., 4)
        One or more quaternions to compare against ``q1``.

    Returns
    -------
    numpy.ndarray
        Angular distance(s) in degrees between ``q1`` and ``q2``.
    """

    q1 = np.asarray(q1, dtype=float).reshape(4)
    q2 = np.asarray(q2, dtype=float).reshape(-1, 4)

    # Normalise quaternions to unit length
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2, axis=1, keepdims=True)

    dots = np.abs(q2 @ q1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = 2.0 * np.arccos(dots) * 180.0 / np.pi
    return angles
