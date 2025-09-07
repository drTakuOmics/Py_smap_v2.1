import numpy as np
from .crop_pad import crop_or_pad

def _has_large_prime_factor(n):
    for p in (2, 3):
        while n % p == 0 and n > 1:
            n //= p
    return n != 1

def _adjust_dim(dim, inc):
    while _has_large_prime_factor(dim):
        dim += inc
    return dim

def resize_for_fft(arr, mode='crop', pad_value=0):
    """Resize array to FFT-friendly dimensions.

    Dimensions are adjusted so their prime factors are only 2 or 3. The final
    array is cubic with the largest adjusted dimension and is centered using
    :func:`crop_or_pad`.

    Parameters
    ----------
    arr : array_like
        Input array.
    mode : {'crop', 'pad'}, optional
        Whether to crop down or pad up to reach an FFT size.
    pad_value : scalar, optional
        Value to use when padding; ignored in ``crop`` mode.

    Returns
    -------
    numpy.ndarray
        Resized array suitable for efficient FFT computation.
    """
    arr = np.asarray(arr)
    inc = -1 if mode == 'crop' else 1
    dims_new = [_adjust_dim(d, inc) for d in arr.shape]
    new_dim = max(dims_new)
    dims_new = [new_dim] * arr.ndim
    mean_val = np.nanmean(arr)
    shifted = arr - mean_val
    result = crop_or_pad(shifted, dims_new, pad_value if mode == 'pad' else 0)
    return result + mean_val
