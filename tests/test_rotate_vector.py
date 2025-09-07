import numpy as np
from smap_tools_python.rotate import rotate3d_vector


def test_rotate3d_vector_shapes():
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    v = np.array([1.0, 0.0, 0.0])
    out = rotate3d_vector(R, v)
    assert np.allclose(out, [0.0, 1.0, 0.0])

    v_rows = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out_rows = rotate3d_vector(R, v_rows)
    assert np.allclose(out_rows, [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

    v_cols = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    out_cols = rotate3d_vector(R, v_cols)
    expected_cols = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 0.0]])
    assert np.allclose(out_cols, expected_cols)
