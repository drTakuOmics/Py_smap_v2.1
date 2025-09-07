import numpy as np
from smap_tools_python import approx_mtf


def test_approx_mtf_center_value():
    params = (1.0, 0.5, 0.2, 1.0, 2.0)
    mtf = approx_mtf(8, params)
    assert mtf.shape == (8, 8)
    expected = params[0] + params[1] + params[2]
    assert np.isclose(mtf[4, 4], expected)


def test_approx_mtf_bin_factor_resize():
    params = (1.0, 0.5, 0.2, 1.0, 2.0)
    mtf = approx_mtf(8, params, bin_factor=2)
    assert mtf.shape == (16, 16)
