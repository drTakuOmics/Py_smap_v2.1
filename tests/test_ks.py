import numpy as np
from smap_tools_python import get_ks


def test_get_ks_basic():
    k, center = get_ks(5, 2.0)
    assert center == 2
    assert k.shape == (5, 5)
    assert k[center, center] == 0
    expected_corner = np.sqrt(2) / 2 / 2.0
    assert np.isclose(k[0, 0], expected_corner)
