import numpy as np


def pairwise_qd(q1, q2=None):
    """Compute pairwise quaternion distances in degrees.

    Parameters
    ----------
    q1 : array_like, shape (..., 4) or (4, N)
        First set of quaternions. The quaternion components may be provided in
        the first dimension or the last dimension.
    q2 : array_like, shape (..., 4) or (4, M), optional
        Optional second set of quaternions. If omitted, pairwise distances
        within ``q1`` are returned with the lower triangle and diagonal set to
        zero to mirror MATLAB's ``pairwiseQD`` behaviour.

    Returns
    -------
    numpy.ndarray
        Matrix of angular distances in degrees with shape ``(N, M)``. When
        ``q2`` is ``None`` the output is ``(N, N)`` with the lower triangle and
        diagonal filled with zeros.
    """

    q1 = np.asarray(q1, dtype=float)
    if q1.shape[0] == 4 and q1.ndim == 2:
        q1 = q1.T
    elif q1.shape[-1] != 4:
        raise ValueError("q1 must have 4 components per quaternion")

    q1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True)

    if q2 is None:
        q2n = q1
    else:
        q2 = np.asarray(q2, dtype=float)
        if q2.shape[0] == 4 and q2.ndim == 2:
            q2 = q2.T
        elif q2.shape[-1] != 4:
            raise ValueError("q2 must have 4 components per quaternion")
        q2n = q2 / np.linalg.norm(q2, axis=1, keepdims=True)

    dots = np.abs(np.clip(q1 @ q2n.T, -1.0, 1.0))
    ang = 2.0 * np.arccos(dots) * 180.0 / np.pi

    if q2 is None:
        ang[np.tril_indices_from(ang)] = 0.0
    return ang
