import numpy as np
from .phase_shift import apply_phase_shifts
from .crop_pad import cutj


def crop_patch_from_image(arr, half_dim, row_col_inds):
    """Extract a square patch centered at ``row_col_inds``.

    The patch is taken from ``arr`` by first shifting the image so that the
    pixel specified by ``row_col_inds`` moves to the image centre, then cropping
    a region of size ``(2*half_dim, 2*half_dim)`` around the centre.  Indices are
    zero-based.

    Parameters
    ----------
    arr : array_like
        Input 2-D image.
    half_dim : int
        Half the side length of the desired patch. The output patch will be of
        size ``2*half_dim``.
    row_col_inds : tuple of int
        Row and column index of the pixel that should be moved to the centre
        before cropping.

    Returns
    -------
    numpy.ndarray
        Extracted image patch.
    """
    arr = np.asarray(arr)
    center = np.array(arr.shape[:2]) // 2
    shifts = center - np.asarray(row_col_inds, dtype=float)
    shifted = apply_phase_shifts(arr, shifts)
    if np.isrealobj(arr):
        shifted = shifted.real
    size = int(half_dim) * 2
    patch = cutj(shifted, (size, size))
    return patch


def crop_patch_from_image3(arr, half_dim, row_col_inds):
    """Extract a centred patch from each frame of a 3‑D stack.

    Parameters
    ----------
    arr : array_like, shape (H, W, N)
        Input stack of ``N`` images.
    half_dim : int
        Half the side length of the desired patch. Output will be ``2*half_dim``
        pixels square for each frame.
    row_col_inds : tuple of float
        Coordinates of the pixel to place at the patch centre before cropping.

    Returns
    -------
    numpy.ndarray
        Stack of cropped patches with shape ``(2*half_dim, 2*half_dim, N)``.
    """

    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError("arr must be a 3-D stack")

    center = np.array(arr.shape[:2]) // 2
    shifts_xy = center - np.asarray(row_col_inds, dtype=float)
    shifts = np.array([shifts_xy[0], shifts_xy[1], 0.0])
    shifted = apply_phase_shifts(arr, shifts)
    if np.isrealobj(arr):
        shifted = shifted.real
    size = int(half_dim) * 2
    return cutj(shifted, (size, size, arr.shape[2]))