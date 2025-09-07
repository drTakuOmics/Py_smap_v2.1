import numpy as np
from .crop_pad import crop_or_pad


def _has_large_prime_factor(n):
    for p in (2, 3):
        while n % p == 0 and n > 1:
            n //= p
    return n != 1


def _prev_good_dim(dim):
    while _has_large_prime_factor(dim):
        dim -= 1
    return dim


def pad_for_fft(arr, method="image"):
    """Crop array to nearest FFT-friendly size.

    The output dimension has no prime factors greater than three, matching
    MATLAB's :code:`padForFFT` which actually crops rather than pads.

    Parameters
    ----------
    arr : array_like
        Input 2‑D array.
    method : {'image', 'filter'}, optional
        Unused but kept for MATLAB parity.

    Returns
    -------
    numpy.ndarray
        Cropped array with dimensions containing only factors 2 and 3.
    """
    arr = np.asarray(arr)
    target = _prev_good_dim(max(arr.shape))
    return crop_or_pad(arr, (target, target), 0)
