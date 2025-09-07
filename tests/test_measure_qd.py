import numpy as np
from smap_tools_python import measure_qd


def test_measure_qd_basic():
    q1 = [1, 0, 0, 0]
    q2 = [np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]
    assert np.isclose(measure_qd(q1, q2)[0], 90.0)


def test_measure_qd_multiple_and_sign():
    q1 = [1, 0, 0, 0]
    q2 = np.array([
        [np.cos(np.pi/4), np.sin(np.pi/4), 0, 0],  # 90° around x
        [-1, 0, 0, 0],                             # same orientation
    ])
    out = measure_qd(q1, q2)
    assert np.allclose(out, [90.0, 0.0])
