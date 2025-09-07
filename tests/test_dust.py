import numpy as np
from smap_tools_python import dust


def test_dust_removes_small_cluster():
    vol = np.zeros((6, 6, 6), dtype=float)
    vol[1:4, 1:4, 1:4] = 10  # large cluster
    vol[5, 5, 5] = 10        # tiny dust
    out = dust(vol, (1, 2))
    assert out[5, 5, 5] < 5  # dust removed
    assert out[2, 2, 2] > 5  # large cluster preserved
