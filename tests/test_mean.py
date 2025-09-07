import numpy as np
from smap_tools_python import mean


def test_mean_nan_ignored():
    arr = np.array([[1.0, np.nan], [3.0, 4.0]])
    np.testing.assert_allclose(mean(arr), [2.0, 4.0])
    np.testing.assert_allclose(mean(arr, axis=1), [1.0, 3.5])
