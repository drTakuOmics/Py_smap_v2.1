import numpy as np
from .cos_mask import rrj


def get_ks(imref, a_per_pix):
    """Compute radial frequency map for an image reference.

    Parameters
    ----------
    imref : array-like or int
        Reference image or its size. If an integer is provided, a square array
        of that size is assumed.
    a_per_pix : float
        Pixel size in Angstroms per pixel.

    Returns
    -------
    k_2d : ndarray
        Radial frequency at each pixel.
    center_pixel : int
        Index of the central pixel (0-based).
    """
    if np.isscalar(imref):
        imref = np.zeros((int(imref), int(imref)), dtype=np.float32)
    Npix = imref.shape[0]
    center_pixel = Npix // 2
    k_2d = rrj(imref.shape) / a_per_pix
    return k_2d, center_pixel
