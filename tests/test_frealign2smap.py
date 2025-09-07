import numpy as np
from smap_tools_python import frealign2smap


def test_frealign2smap_zero():
    R, q = frealign2smap([[0, 0, 0]])
    R = R[:, :, 0]
    q = q[:, 0]
    expected_R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    expected_q = np.array([0.0, 1.0, 0.0, 0.0])
    assert np.allclose(R, expected_R)
    assert np.allclose(q, expected_q)
