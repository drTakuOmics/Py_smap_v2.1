import numpy as np
def rotate2d_matrix(image, R):
    """Rotate a 2-D array using a rotation matrix.

    The implementation performs nearest-neighbour interpolation purely with
    NumPy so that it has no heavy dependencies.  It is therefore limited to
    moderate rotations but suffices to mirror the basic behaviour of the
    original MATLAB ``rotate2dMatrix`` utility.

    Parameters
    ----------
    image : array_like, shape (M, N)
        2-D image to rotate.
    R : array_like, shape (2, 2) or (3, 3)
        Rotation matrix. If a 3×3 matrix is supplied, the upper-left 2×2 block
        is used.

    Returns
    -------
    numpy.ndarray
        Rotated image with the same shape as the input.
    """

    image = np.asarray(image)
    R = np.asarray(R, dtype=float)
    if R.shape == (3, 3):
        R = R[:2, :2]
    # handle pure 90° rotations via rot90j for exactness
    rot90_mats = {
        1: np.array([[0, -1], [1, 0]]),
        2: np.array([[-1, 0], [0, -1]]),
        3: np.array([[0, 1], [-1, 0]]),
    }
    for k, mat in rot90_mats.items():
        if np.allclose(R, mat):
            return rot90j(image, k)

    n = np.array(image.shape)
    center = (n - 1) / 2.0
    y, x = np.indices(n)
    coords = np.stack((x.ravel(), y.ravel()), axis=1)
    coords = coords - center[::-1]
    coords = coords @ R.T + center[::-1]
    coords = np.rint(coords).astype(int)
    mask = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < n[1])
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < n[0])
    )
    out = np.zeros_like(image)
    out[y.ravel()[mask], x.ravel()[mask]] = image[coords[mask, 1], coords[mask, 0]]
    return out


def rotate3d_matrix(volume, R):
    """Rotate a 3-D volume using a rotation matrix.

    As with :func:`rotate2d_matrix`, nearest-neighbour interpolation is used to
    avoid external dependencies.  The function preserves the input shape and
    fills voxels falling outside the rotated volume with zeros.

    Parameters
    ----------
    volume : array_like, shape (X, Y, Z)
        3-D volume to rotate.
    R : array_like, shape (3, 3)
        Rotation matrix.

    Returns
    -------
    numpy.ndarray
        Rotated volume with the same shape as the input.
    """

    volume = np.asarray(volume)
    R = np.asarray(R, dtype=float)
    if np.allclose(R, np.eye(3)):
        return volume.copy()
    rot_z90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    if np.allclose(R, rot_z90):
        out = np.rot90(volume, 1, axes=(1, 0))
        out = np.roll(out, 1, axis=0)
        return out

    n = np.array(volume.shape)
    center = (n - 1) / 2.0
    z, y, x = np.indices(n)
    coords = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
    coords = coords - center[::-1]
    coords = coords @ R.T + center[::-1]
    coords = np.rint(coords).astype(int)
    mask = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < n[2])
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < n[1])
        & (coords[:, 2] >= 0)
        & (coords[:, 2] < n[0])
    )
    out = np.zeros_like(volume)
    out[
        z.ravel()[mask],
        y.ravel()[mask],
        x.ravel()[mask],
    ] = volume[coords[mask, 2], coords[mask, 1], coords[mask, 0]]
    return out

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