import numpy as np
from .fft import ftj
from .crop_pad import extendj


def occ(imref, templateref, q2=None):
    """Compute occupancy cross-correlation between image and template.

    Parameters
    ----------
    imref : array_like
        Input image (real or already Fourier transformed).
    templateref : array_like
        Template image (real or Fourier domain). If real and smaller than
        ``imref`` it will be padded to match dimensions.
    q2 : array_like, optional
        Weighting array; defaults to ones with the shape of ``imref``.

    Returns
    -------
    numpy.ndarray
        Real-valued cross-correlation map scaled by template power.
    """
    imref = np.asarray(imref)
    templateref = np.asarray(templateref)

    imref_F = ftj(imref) if np.isrealobj(imref) else imref

    if np.isrealobj(templateref):
        template = templateref - np.median(templateref)
        if template.shape != imref.shape:
            template = extendj(template, imref.shape, pad_value=0)
        templateref_F = ftj(template)
    else:
        templateref_F = templateref

    if q2 is None:
        q2 = np.ones_like(imref, dtype=float)
    else:
        q2 = np.asarray(q2)

    N = imref.shape[0]
    n_dims = imref.ndim
    pix_factor = np.sqrt(N ** n_dims)

    denom_F = np.abs(templateref_F) ** 2
    denom_F_sum = np.sum(q2 * denom_F) / pix_factor

    cc_F = imref_F * np.conj(templateref_F)
    cc = np.real(np.fft.ifftn(np.fft.fftshift(cc_F)))
    cc = np.fft.ifftshift(cc)

    return cc * (pix_factor / denom_F_sum)
