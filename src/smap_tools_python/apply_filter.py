import numpy as np
from .fft import ftj, iftj
from .crop_pad import crop_or_pad


def apply_filter(imref, tm, norm=True):
    """Apply Fourier-domain filter to image or volume.

    Parameters
    ----------
    imref : array_like
        Input 2-D or 3-D array.
    tm : array_like
        Filter in Fourier space (same shape as ``imref`` or will be
        center-cropped/extended to match).
    norm : bool, optional
        If ``True``, normalize the input to zero mean and unit variance
        before filtering.
    """
    imref = np.asarray(imref, dtype=float)
    tm = np.asarray(tm, dtype=float)
    in_dims = imref.shape

    if norm:
        mean = np.mean(imref)
        std = np.std(imref)
        imref = (imref - mean) / (std if std else 1)

    if imref.shape != tm.shape:
        imref = crop_or_pad(imref, tm.shape, np.mean(imref))

    im_f = ftj(imref)
    out = iftj(im_f * tm)
    return crop_or_pad(out, in_dims, 0)
