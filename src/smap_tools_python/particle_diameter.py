import numpy as np

from .cos_mask import rrj
from .bindata import bindata


def particle_diameter(vol, thresh=0.005):
    """Estimate particle diameter from a 3-D volume.

    Parameters
    ----------
    vol : array_like
        Input 3-D volume.
    thresh : float, optional
        Fraction of the maximal radial profile used to determine the
        diameter. Defaults to 0.005, matching the MATLAB implementation.

    Returns
    -------
    float
        Estimated diameter in pixels.
    """
    vol = np.asarray(vol)
    Npix = max(vol.shape)
    r_coord = rrj(vol.shape) * Npix

    if Npix % 2:
        r_bins = np.linspace(0, np.sqrt(2) / 2, int(Npix * np.sqrt(2) / 2) + 1)
    else:
        r_bins = np.linspace(0, np.sqrt(2) / 2, int((Npix + 1) * np.sqrt(2) / 2) + 1)[:-1]
    r_bins *= Npix

    r_coord = r_coord.ravel()
    vol_flat = vol.astype(float).ravel()
    binned, *_ = bindata(vol_flat, r_coord, r_bins)
    binned = binned - np.median(binned)
    binned /= np.max(binned)
    idx = np.where(binned > thresh)[0]
    if idx.size == 0:
        return 0.0
    return float(idx[-1] * 2)
