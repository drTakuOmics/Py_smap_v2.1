import numpy as np


def rotate3d_vector(R, v):
 
def normalize_rotation_matrices(R):
    """Project matrices onto the closest proper rotation matrices.

    Each input matrix is orthonormalized via singular value decomposition
    following the approach of MATLAB's ``normalizeRM`` helper.  The operation
    preserves orientation (determinant ``+1``) and can handle a single matrix
    or a stack of matrices provided in the final two dimensions.

    Parameters
    ----------
    R : array_like, shape (..., 3, 3)
        One or more rotation matrices to be normalized.

    Returns
    -------
    numpy.ndarray
        The normalized rotation matrices with the same shape as ``R``.
    """

    R = np.asarray(R, dtype=float)
    if R.shape[-2:] != (3, 3):
        raise ValueError("Input must have shape (...,3,3)")
    orig_shape = R.shape
    mats = R.reshape(-1, 3, 3)
    U, _, Vt = np.linalg.svd(mats)
    Rn = np.matmul(U, Vt)
    # ensure det +1
    det = np.linalg.det(Rn)
    neg = det < 0
    if np.any(neg):
        U[neg, :, 2] *= -1
        Rn = np.matmul(U, Vt)
    return Rn.reshape(orig_shape)