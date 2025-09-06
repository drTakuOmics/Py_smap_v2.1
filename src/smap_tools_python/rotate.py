def rotate3d_vector(R, v):
    """Rotate 3-D vectors using a rotation matrix.

    Parameters
    ----------
    R : array_like, shape (3, 3)
        Rotation matrix.
    v : array_like, shape (..., 3) or (3, ...)
        Vectors to rotate. If the leading dimension is not 3, the function
        assumes vectors are provided row-wise.

    Returns
    -------
    numpy.ndarray
        Rotated vector(s) with the same orientation as ``v``.
    """
    import numpy as np

    R = np.asarray(R)
    v = np.asarray(v)
    if v.ndim == 1:
        return R @ v
    if v.shape[0] != 3:
        return (R @ v.T).T
    return R @ v
