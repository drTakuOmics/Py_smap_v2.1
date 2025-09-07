"""Parameterize electron scattering factors for common atoms."""
from __future__ import annotations

import numpy as np

# Subset of coefficients from Peng et al. (1996) used by the MATLAB
# ``parameterizeSF`` routine. Values are split into ``a`` and ``b``
# components (first five and last five respectively).
_COEFFS = {
    "H": [0.0349, 0.1201, 0.1970, 0.0573, 0.1195, 0.5347, 3.5867, 12.3471, 18.9525, 38.6269],
    "HE": [0.0317, 0.0838, 0.1526, 0.1334, 0.0164, 0.2507, 1.4751, 4.4938, 12.6646, 31.1653],
    "C": [0.0893, 0.2563, 0.7570, 1.0487, 0.3575, 0.2465, 1.7100, 6.4094, 18.6113, 50.2523],
    "N": [0.1022, 0.3219, 0.7982, 0.8197, 0.1715, 0.2451, 1.7481, 6.1925, 17.3894, 48.1431],
    "O": [0.0974, 0.2921, 0.6910, 0.6990, 0.2039, 0.2067, 1.3815, 4.6943, 12.7105, 32.4726],
    "F": [0.1083, 0.3175, 0.6487, 0.5846, 0.1421, 0.2057, 1.3439, 4.2788, 11.3932, 28.7881],
    "P": [0.2548, 0.6106, 1.4541, 2.3204, 0.8477, 0.2908, 1.8740, 8.5176, 24.3434, 63.2996],
    "S": [0.2497, 0.5628, 1.3899, 2.1865, 0.7715, 0.2681, 1.6711, 7.0267, 19.5377, 50.3888],
    "CL": [0.2443, 0.5397, 1.3919, 2.0197, 0.6621, 0.2468, 1.5242, 6.1537, 16.6687, 42.3086],
    "FE": [0.3946, 1.2725, 1.7031, 2.3140, 1.4795, 0.2717, 2.0443, 7.6007, 29.9714, 86.2265],
    "NI": [0.3860, 1.1765, 1.5451, 2.0730, 1.3814, 0.2478, 1.7660, 6.3107, 25.2204, 74.3146],
    "CU": [0.4314, 1.3208, 1.5236, 1.4671, 0.8562, 0.2694, 1.9223, 7.3474, 28.9892, 90.6246],
    "ZN": [0.4288, 1.2646, 1.4472, 1.8294, 1.0934, 0.2593, 1.7998, 6.7500, 25.5860, 73.5284],
}


def parameterize_sf(elem: str = "C") -> tuple[np.ndarray, np.ndarray]:
    """Return Gaussian coefficients for ``elem``.

    Parameters
    ----------
    elem : str, optional
        Element symbol. Defaults to carbon.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Tuple of ``a`` and ``b`` coefficient arrays. Unknown elements yield
        zero arrays and a warning message, mirroring the MATLAB behaviour.
    """

    key = elem.strip().upper()
    coeffs = _COEFFS.get(key)
    if coeffs is None:
        print(f"did not find atom type {elem}")
        return np.zeros(5), np.zeros(5)
    coeffs = np.asarray(coeffs, dtype=float)
    return coeffs[:5], coeffs[5:]
