import numpy as np
from .quaternion import Quaternion
from .rotate import normalize_rotation_matrices


def _to_rotation_matrices(q):
    """Convert various rotation representations to a (3,3,N) array."""
    if isinstance(q, Quaternion):
        return q.to_rotation_matrix()[:, :, np.newaxis]
    arr = np.asarray(q, dtype=float)
    if arr.ndim == 2 and arr.shape == (3, 3):
        return arr[:, :, np.newaxis]
    if arr.ndim == 3 and arr.shape[0:2] == (3, 3):
        return arr
    if arr.shape[-1] == 4:
        arr = arr.reshape(-1, 4)
        mats = np.stack([Quaternion(*row).to_rotation_matrix() for row in arr], axis=2)
        return mats
    raise ValueError("Unsupported rotation representation")


def bump_q(q_in, q_bump):
    """Compose two sets of rotations.

    Parameters
    ----------
    q_in : array_like or Quaternion or sequence
        Initial rotations. Can be rotation matrices of shape ``(3,3,N)`` or
        quaternions of shape ``(N,4)`` or ``(4,N)``.
    q_bump : array_like or Quaternion or sequence
        Rotations to apply after ``q_in``.

    Returns
    -------
    numpy.ndarray
        Rotation matrices for all combinations of ``q_bump * q_in`` with
        shape ``(3,3,N_in*N_bump)``.
    """
    R_in = _to_rotation_matrices(q_in)
    R_bump = _to_rotation_matrices(q_bump)
    n_in = R_in.shape[2]
    n_bump = R_bump.shape[2]
    out = np.empty((3, 3, n_in * n_bump), dtype=float)
    idx = 0
    for i in range(n_in):
        for j in range(n_bump):
            out[:, :, idx] = R_bump[:, :, j] @ R_in[:, :, i]
            idx += 1
    return normalize_rotation_matrices(np.transpose(out,(2,0,1))).transpose(1,2,0)
