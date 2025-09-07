import numpy as np
from smap_tools_python.rrj import rrj


def test_rrj_shape_and_norm():
    arr = np.zeros((5, 7, 3))
    R = rrj(arr)
    assert R.shape == arr.shape
    halves = np.array(arr.shape) // 2
    expected_max = np.sqrt((halves ** 2).sum()) / (2 * (arr.shape[1] // 2))
    assert np.isclose(R.max(), expected_max)
    assert np.isclose(R.min(), 0.0)


def test_rrj_tuple_input():
    R = rrj((4, 4))
    assert R.shape == (4, 4)
    assert np.isclose(R[1, 2], R[2, 1])
    assert R[1, 1] > R[2, 2]
