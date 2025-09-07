import numpy as np
from smap_tools_python import pairwise_qd


def test_pairwise_qd_self():
    qs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]).T  # shape (4,2)
    d = pairwise_qd(qs)
    assert d.shape == (2, 2)
    assert d[0, 1] == 180
    assert np.allclose(d[1, 0], 0)
    assert np.allclose(np.diag(d), 0)


def test_pairwise_qd_two_sets():
    q1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]).T
    q2 = np.array([[0, 0, 1, 0], [0, 0, 0, 1]]).T
    d = pairwise_qd(q1, q2)
    assert d.shape == (2, 2)
    assert np.all(d >= 0)
