import numpy as np
from .crop_patch import crop_patch_from_image
from .resize_f import resize_F


def max_interp_f(imref, half_width=None, interp_factor=4, center=None):
    """Locate image peak with subpixel precision via Fourier interpolation.

    Parameters
    ----------
    imref : ndarray
        Input 2-D image.
    half_width : int, optional
        Half-width of the search patch around the image centre. Defaults to the
        smaller image dimension.
    interp_factor : int, optional
        Integer factor by which to up-sample the patch for subpixel
        localisation.
    center : tuple of int, optional
        If provided, skip the coarse search and assume the peak is already at
        ``center`` (row, col) in the initial patch.

    Returns
    -------
    shift : ndarray, shape (2,)
        Estimated translation ``(dy, dx)`` required to move the peak to the
        image centre.
    peak_value : float
        Maximum value after interpolation.
    """
    imref = np.asarray(imref, dtype=float)
    edge = int(half_width or min(imref.shape))
    cp_init = np.array(imref.shape[:2]) // 2
    cc = crop_patch_from_image(imref, edge, cp_init)
    cp = np.array(cc.shape) // 2

    if center is None:
        ny, nx = np.unravel_index(np.argmax(cc), cc.shape)
    else:
        ny, nx = center

    shifts_fullpix = np.array([ny - cp[0], nx - cp[1]], dtype=float)
    cc_patch = crop_patch_from_image(cc, edge, (ny, nx))
    cc_patch_i = resize_F(cc_patch, interp_factor, method="newSize")
    cp_i = np.array(cc_patch_i.shape) // 2
    ny_i, nx_i = np.unravel_index(np.argmax(cc_patch_i), cc_patch_i.shape)
    shifts_subpix = (np.array([ny_i - cp_i[0], nx_i - cp_i[1]], dtype=float) /
                     interp_factor)
    shifts = shifts_fullpix + shifts_subpix
    shift_vec = -shifts  # move peak to centre
    peak_val = float(cc_patch_i.max())
    return shift_vec, peak_val
