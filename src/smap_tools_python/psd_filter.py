"""Radial power spectrum density filter for 2-D images.

This module implements a lightweight analogue of the MATLAB ``psdFilter``
routine.  The function estimates the isotropic noise background of an image in
Fourier space and returns a whitening filter alongside the filtered image and
the background estimate.  Advanced options of the original code (such as sector
averaging) are omitted for brevity but the core behaviour is preserved.
"""

from __future__ import annotations

import numpy as np

from .fft import ftj, iftj
from .cos_mask import rrj
from .bindata import bindata
from .get_psd import get_psd
from .crop_pad import crop_or_pad
from .nm import nm


def psd_filter(nf_im: np.ndarray, method: str = "psd", n_sectors: int | None = None):
    """Whiten an image based on its radial power spectrum density.

    Parameters
    ----------
    nf_im : ndarray
        Input image.  Any non-square image is padded/cropped to a square prior
        to processing.
    method : {"psd", "sqrt"}, optional
        Determines whether the filter is based on the power spectrum (default)
        or its square root.
    n_sectors : int, optional
        Present for API compatibility but currently unused.

    Returns
    -------
    filter_out : ndarray
        Whitening filter in Fourier space.
    im_out : ndarray
        Filtered image in real space.
    psbg : ndarray
        Estimated radial background of the Fourier amplitude.
    """

    nf_im = np.asarray(nf_im, dtype=float)
    nf_im = nf_im - np.mean(nf_im)

    # work on a square canvas to mirror MATLAB behaviour
    orig_shape = nf_im.shape
    edge = max(orig_shape)
    nf_im = crop_or_pad(nf_im, (edge, edge), 0)

    if method == "psd":
        psd = get_psd(nf_im)
    elif method == "sqrt":
        psd = np.sqrt(get_psd(nf_im))
    else:
        raise ValueError("method must be 'psd' or 'sqrt'")

    cp = edge // 2
    r_coord = rrj(psd.shape) * edge
    inds = r_coord <= cp

    if edge % 2:  # odd
        r_bins = np.arange(0, edge // 2 + 2) / (edge - 1)
    else:
        r_bins = np.linspace(0, 0.5, edge // 2 + 1)
    r_bins *= edge

    _, yb, _ = bindata(psd[inds].ravel(), r_coord[inds].ravel(), r_bins)
    temp = np.full_like(psd, np.nan, dtype=float)
    temp[inds] = yb

    shell = (cp - r_coord < 2) & (cp - r_coord >= 0)
    temp[r_coord > cp] = np.nanmean(temp[shell])
    temp[cp, cp] = 1.0

    filter_out = 1.0 / temp
    filter_out[cp, cp] = 0.0

    im_out = nm(iftj(ftj(nf_im) * filter_out))

    # restore original dimensions
    filter_out = crop_or_pad(filter_out, orig_shape)
    im_out = crop_or_pad(im_out, orig_shape)
    psbg = crop_or_pad(temp, orig_shape)

    return filter_out, im_out, psbg

