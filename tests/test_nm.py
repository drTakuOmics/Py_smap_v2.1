import numpy as np
from smap_tools_python import nm


def test_nm_basic():
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    out = nm(arr)
    assert np.allclose(np.nanmean(out), 0.0)
    assert np.allclose(np.nanstd(out, ddof=1), 1.0)


def test_nm_nan():
    arr = np.array([1.0, np.nan, 3.0])
    out = nm(arr)
    assert np.isnan(out[1])
    expected = np.array([-1 / np.sqrt(2), np.nan, 1 / np.sqrt(2)])
    assert np.allclose(out, expected, equal_nan=True)
