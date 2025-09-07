import numpy as np
from .rotate import normalize_rotation_matrices


def normalize_rm(R):
    """Normalize rotation matrices to be orthonormal with det=+1.

    This is a light-weight wrapper around :func:`normalize_rotation_matrices`
    to mirror MATLAB's ``normalizeRM`` helper.  The input may be a single
    3×3 matrix, an ``(N,3,3)`` stack, or a ``(3,3,N)`` stack.
    """
    arr = np.asarray(R, dtype=float)
    if arr.ndim == 2:
        return normalize_rotation_matrices(arr)
    if arr.shape[:2] == (3, 3):
        arr_t = arr.transpose(2, 0, 1)
        return normalize_rotation_matrices(arr_t).transpose(1, 2, 0)
    if arr.shape[-2:] == (3, 3):
        return normalize_rotation_matrices(arr)
    raise ValueError("Input must be a 3x3 matrix or stack of such matrices")
