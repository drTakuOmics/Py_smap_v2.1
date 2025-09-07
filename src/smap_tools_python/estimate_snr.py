"""Signal-to-noise ratio estimate from molecular weight and thickness.

This implements the lightweight empirical model used in SMAP's MATLAB
``estimate_SNR`` helper.  The formulation is:

SNR = (a2 + a1 * sqrt(MW)) * exp(-thickness / lambda)

where ``MW`` is the molecular weight in kilodaltons and ``thickness`` is
in nanometres.  The constants are derived from fits to experimental data.
The function accepts scalar or array inputs and broadcasts parameters
accordingly.
"""

from __future__ import annotations

import numpy as np

_A1 = 0.4654
_A2 = 2.0978
_LAMBDA = 426.0


def estimate_snr(mw_kda, thickness_nm, lambda_nm: float = _LAMBDA):
    """Estimate SNR for a particle of given size and sample thickness.

    Parameters
    ----------
    mw_kda : array_like
        Molecular weight of the particle in kilodaltons.
    thickness_nm : array_like
        Sample thickness in nanometres.
    lambda_nm : float, optional
        Attenuation length in nanometres.  Defaults to the empirically
        determined value of 426 nm used by SMAP.

    Returns
    -------
    numpy.ndarray
        Estimated signal-to-noise ratio.
    """

    mw_kda = np.asarray(mw_kda, dtype=float)
    thickness_nm = np.asarray(thickness_nm, dtype=float)
    return (_A2 + _A1 * np.sqrt(mw_kda)) * np.exp(-thickness_nm / lambda_nm)
