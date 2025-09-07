import numpy as np

from smap_tools_python import bindata


def test_bindata_1d():
    y = np.array([1, 2, 3, 4])
    x = np.array([0.1, 0.2, 0.8, 0.9])
    edges = [0.0, 0.5, 1.0]
    ym, yb, y_full = bindata(y, x, edges)
    assert np.allclose(ym, [1.5, 3.5], equal_nan=False)
    assert np.allclose(yb, [1.5, 1.5, 3.5, 3.5])
    assert np.allclose(y_full, [0.0, 0.5])
