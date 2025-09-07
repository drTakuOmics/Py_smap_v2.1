from __future__ import annotations

import numpy as np

from .fft import ftj, iftj
from .polar_image import polar_image


def ipcc(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Phase correlation in log-polar space.

    This is a lightweight translation of SMAP's ``ipcc`` MATLAB
    routine. ``image`` and ``template`` are assumed to be Fourier
    magnitude images. They are warped into log-polar coordinates, and the
    phase correlation of the polar representations is computed to
    estimate in-plane rotation and isotropic scaling.

    Parameters
    ----------
    image, template : ndarray
        Real-valued 2-D arrays of identical shape.

    Returns
    -------
    ndarray
        The log-polar cross-correlation map. The rotation can be
        determined from the column index of the peak location.
    """
    if image.shape != template.shape:
        raise ValueError("image and template must have the same shape")

    n = image.shape[0]
    # Convert to log-polar coordinates around the image centre
    img_p = polar_image(image, n, n, method="bicubic", shape="valid")[0]
    tmpl_p = polar_image(template, n, n, method="bicubic", shape="valid")[0]

    # Normalize each polar image
    img_p = (img_p - np.mean(img_p)) / np.std(img_p)
    tmpl_p = (tmpl_p - np.mean(tmpl_p)) / np.std(tmpl_p)

    # Phase correlation
    cc = ftj(img_p) * np.conj(ftj(tmpl_p))
    out = iftj(cc)

    # Normalize rows to unit variance as in the MATLAB implementation
    out -= out.mean(axis=1, keepdims=True)
    out /= out.std(axis=1, keepdims=True)
    return out


# MATLAB-style alias
ipcc_m = ipcc
