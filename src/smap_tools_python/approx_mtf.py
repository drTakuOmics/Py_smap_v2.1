import numpy as np
from .ks import get_ks
from scipy.ndimage import zoom


def approx_mtf(size, params, bin_factor=1):
    """Approximate the microscope transfer function.

    Parameters
    ----------
    size : int
        Linear dimension of the (square) output array.
    params : sequence of float
        Parameters ``(a, b, c, alpha, beta)`` controlling the model.
    bin_factor : float, optional
        Optional scaling factor. Values other than ``1`` trigger a Fourier
        resize using :func:`resize_for_fft` to maintain fixed output size.

    Returns
    -------
    numpy.ndarray
        Approximate modulation transfer function values on a square grid.
    """
    a, b, c, alpha, beta = params
    k_2d, _ = get_ks(size, 0.5)
    mtf = (a / (1 + alpha * k_2d**2)) + (b / (1 + beta * k_2d**2)) + c
    if bin_factor != 1:
        mtf = zoom(mtf, bin_factor, order=1)
        new_size = int(np.round(size * bin_factor))
        mtf = mtf[:new_size, :new_size]
    return mtf
