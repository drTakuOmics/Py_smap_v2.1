import numpy as np


def mtf_mm(params, k):
    """Modulation transfer function model from McMullan et al. 2014.

    Parameters
    ----------
    params : sequence of float
        Concatenated ``a`` and ``lambda`` values. ``a`` specifies the amplitude
        coefficients and ``lambda`` the decay lengths.
    k : array_like
        Spatial frequency at which to evaluate the MTF.

    Returns
    -------
    numpy.ndarray
        The MTF evaluated at ``k``.
    """

    params = np.asarray(params, dtype=float)
    k = np.asarray(k, dtype=float)
    half = params.size // 2
    a = params[:half]
    lam = params[half:]
    F = np.sinc(k / 2.0)
    for ai, li in zip(a, lam):
        F = F + ai * np.exp(-(np.pi ** 2) * (li ** 2) * (k ** 2) / 4.0)
    return F
