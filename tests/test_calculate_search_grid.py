import numpy as np
from smap_tools_python import calculate_search_grid


def test_calculate_search_grid_c1():
    RM, EA = calculate_search_grid('C1', 90, 180)
    assert EA.shape[1] == 14
    assert RM.shape == (3, 3, 28)
    assert np.allclose(RM[:, :, 0], np.eye(3))
    # ensure orthogonality of a sample matrix
    idx = RM.shape[2] - 1
    RtR = RM[:, :, idx].T @ RM[:, :, idx]
    assert np.allclose(RtR, np.eye(3), atol=1e-6)
