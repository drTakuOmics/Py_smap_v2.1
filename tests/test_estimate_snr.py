import numpy as np
from smap_tools_python import estimate_snr


def test_estimate_snr_values():
    # Reference values computed from MATLAB implementation
    mw = np.array([200, 800])
    thickness = np.array([100, 50])
    out = estimate_snr(mw, thickness)
    expected = (2.0978 + 0.4654 * np.sqrt(mw)) * np.exp(-thickness / 426.0)
    assert np.allclose(out, expected)
