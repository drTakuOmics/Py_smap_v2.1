import numpy as np
from .fft import ftj, iftj


def mask_central_cross(imref):
    """Zero out the central cross in Fourier space."""
    Npix = imref.shape[0]
    cp = Npix // 2
    if np.isrealobj(imref):
        imref_F = ftj(imref)
        ift_flag = True
    else:
        imref_F = imref
        ift_flag = False
    dc_val = np.abs(imref_F[cp, cp])
    imref_F[cp, :] = 0
    imref_F[:, cp] = 0
    imref_F[cp, cp] = dc_val
    if ift_flag:
        return iftj(imref_F)
    return imref_F
