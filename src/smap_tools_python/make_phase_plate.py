import numpy as np
from .cos_mask import rrj
from .fft import ftj, iftj
from .make_filt import make_filt


def make_phase_plate(imref, method="vulovic", k_cuton=None):
    """Construct a phase plate transfer function.

    Parameters
    ----------
    imref : array_like
        Reference image whose first dimension sets the output size.
    method : {"vulovic", "denk"}, optional
        Strategy for constructing the phase plate.
    k_cuton : float, optional
        Radius in pixels at which the phase plate transitions. Defaults to
        ``Npix/100`` for ``vulovic`` and ``Npix/50`` for ``denk``.

    Returns
    -------
    numpy.ndarray
        Complex-valued phase plate of shape ``(Npix, Npix)``.
    """
    imref = np.asarray(imref)
    Npix = imref.shape[0]
    if k_cuton is None:
        k_cuton = Npix / (100 if method == "vulovic" else 50)

    k2d = rrj((Npix, Npix)) * Npix
    pp = np.ones((Npix, Npix), dtype=np.complex64) * np.exp(1j * np.pi / 2)
    pp[k2d <= k_cuton] = 1

    mask = make_filt(Npix, ((1.0 / k_cuton) * 0.5, (1.0 / k_cuton) * 0.8), 0.25)

    if method == "vulovic":
        pp_re = iftj(ftj(pp.real)) * mask
        pp_im = 1 - pp_re
    elif method == "denk":
        pp_im = iftj(ftj(1 - pp.imag)) * mask
        pp_re = 1 - pp_im
    else:
        raise ValueError("Unsupported method")

    return pp_re + 1j * pp_im
