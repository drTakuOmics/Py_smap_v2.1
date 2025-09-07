import numpy as np


def q2r(q):
    """Convert quaternion(s) to rotation matrix/matrices.

    Parameters
    ----------
    q : array_like, shape (..., 4)
        Quaternion(s) in ``w, x, y, z`` order. Any leading dimensions are
        interpreted as batch dimensions.

    Returns
    -------
    numpy.ndarray
        Array of rotation matrices with shape ``(..., 3, 3)``.
    """
    q = np.asarray(q, dtype=float)
    q = np.atleast_2d(q)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n[n == 0] = 1
    q = q / n
    w, x, y, z = q.T
    R = np.empty(q.shape[:-1] + (3, 3), dtype=float)
    R[..., 0, 0] = w**2 + x**2 - y**2 - z**2
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = w**2 - x**2 + y**2 - z**2
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = w**2 - x**2 - y**2 + z**2
    # Orthonormalize using SVD to counter numerical drift
    u, _, v = np.linalg.svd(R)
    R = u @ v
    return R[0] if R.shape[0] == 1 else R
