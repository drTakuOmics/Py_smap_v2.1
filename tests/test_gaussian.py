import numpy as np
from smap_tools_python import g2


def test_g2_scalar():
    assert np.isclose(g2(0.0), 1.0)
    val = g2(1.0, (2.0, 1.0))
    assert np.isclose(val, 2.0 * np.exp(-0.5))


def test_g2_array():
    x = np.array([-1.0, 0.0, 1.0])
    res = g2(x, (1.0, 1.0))
    expected = np.exp(-x**2 / 2.0)
    assert np.allclose(res, expected)
