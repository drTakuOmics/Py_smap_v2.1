import numpy as np
from smap_tools_python import get_psd


def test_get_psd_delta():
    im = np.zeros((4, 4), dtype=float)
    im[2, 2] = 1.0
    psd = get_psd(im)
    expected = np.full((4, 4), 1 / 16)
    expected[2, 2] = 0
    assert np.allclose(psd, expected)
