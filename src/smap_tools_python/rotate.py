import numpy as np


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







def rot90j(arr, k=0):
    """Rotate an array by 90° increments while keeping its center aligned.

    This mirrors the MATLAB ``rot90j`` helper, applying the same pixel shifts
    for even-sized arrays so that the central pixel remains in place after
    rotation.

    Parameters
    ----------
    arr : array_like
        2-D array to rotate.
    k : int, optional
        Number of 90° rotations. Positive values rotate counter-clockwise.

    Returns
    -------
    numpy.ndarray
        Rotated array.
    """

    k = int(k) % 4
    if k == 0:
        return np.array(arr, copy=True)

    out = np.rot90(arr, k)
    edge = out.shape[0]
    if edge % 2 == 0:
        if k == 1:
            shifts = (1, 0)
        elif k == 2:
            shifts = (1, 1)
        elif k == 3:
            shifts = (0, 1)
        else:
            shifts = (0, 0)
        out = np.roll(out, shifts, axis=(0, 1))
    return out


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
    n = np.array(image.shape)
    center = (n - 1) / 2.0
    # coordinates of output pixels
    grid = np.indices(n).reshape(2, -1)
    coords = grid.T - center
    coords = coords @ R.T + center
    coords = np.rint(coords).astype(int)
    mask = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < n[0])
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < n[1])
    )
    out = np.zeros_like(image)
    out[grid[0, mask], grid[1, mask]] = image[coords[mask, 0], coords[mask, 1]]
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
    n = np.array(volume.shape)
    center = (n - 1) / 2.0
    grid = np.indices(n).reshape(3, -1)
    coords = grid.T - center
    coords = coords @ R.T + center
    coords = np.rint(coords).astype(int)
    mask = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < n[0])
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < n[1])
        & (coords[:, 2] >= 0)
        & (coords[:, 2] < n[2])
    )
    out = np.zeros_like(volume)
    out[
        grid[0, mask],
        grid[1, mask],
        grid[2, mask],
    ] = volume[coords[mask, 0], coords[mask, 1], coords[mask, 2]]
   


    return out