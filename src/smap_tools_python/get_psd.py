import numpy as np

from .getcp import getcp


def get_psd(image):
    """Compute the power spectral density of ``image``.

    Parameters
    ----------
    image : array_like
        Real-space image whose PSD is desired. ``NaN`` values are treated as
        zeros.

    Returns
    -------
    numpy.ndarray
        Power spectral density scaled by ``n^2`` with the DC component set to
        zero, matching the MATLAB ``getPSD`` helper.
    """

    im = np.asarray(image, dtype=float)
    n = im.shape[0]
    nf = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(np.nan_to_num(im))))
    psd = np.abs(nf) ** 2
    cp = getcp(im)
    psd[int(cp[0]), int(cp[1])] = 0
    return psd / (n ** 2)
