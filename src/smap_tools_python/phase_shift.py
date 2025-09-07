import numpy as np
from numpy.fft import fftn, ifftn
from scipy.ndimage import fourier_shift


def apply_phase_shifts(arr, shifts):
    """Shift an array by subpixel amounts using Fourier phase shifts.

    Parameters
    ----------
    arr : array_like
        Input array to shift. Can be N-dimensional.
    shifts : array_like
        Sequence of length ``arr.ndim`` giving shifts along each axis in
        pixels. Positive values shift toward higher indices.

    Returns
    -------
    numpy.ndarray
        The shifted array, potentially complex when subpixel shifts are used.
    """
    arr = np.asarray(arr)
    shifts = np.asarray(shifts, dtype=float)
    if shifts.shape != (arr.ndim,):
        raise ValueError("shifts must match the number of array dimensions")
    shifted_fft = fourier_shift(fftn(arr), shifts)
    return ifftn(shifted_fft)
