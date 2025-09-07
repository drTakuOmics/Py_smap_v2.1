"""Radial power spectrum density filter for 3-D volumes."""

from __future__ import annotations

import numpy as np

from .cos_mask import rrj
from .bindata import bindata


def psd_filter_3d(inref: np.ndarray) -> np.ndarray:
    """Generate a 3-D whitening filter from a volume's power spectrum."""

    inref = np.asarray(inref, dtype=float)
    Npix = inref.shape[0]
    cp = Npix // 2

    r_coord = rrj(inref.shape) * Npix
    inds = r_coord <= cp

    if Npix % 2:
        r_bins = np.arange(0, Npix // 2 + 2) / (Npix - 1)
    else:
        r_bins = np.linspace(0, 0.5, Npix // 2 + 1)
    r_bins *= Npix

    _, yb, _ = bindata(inref[inds].ravel(), r_coord[inds].ravel(), r_bins)
    temp = np.full((Npix, Npix, Npix), np.nan, dtype=float)
    temp[inds] = yb

    shell = (cp - r_coord < 2) & (cp - r_coord >= 0)
    temp[r_coord > cp] = np.nanmean(temp[shell])
    temp[cp, cp, cp] = 1.0

    filt = 1.0 / temp
    filt[cp, cp, cp] = 0.0
    return filt

