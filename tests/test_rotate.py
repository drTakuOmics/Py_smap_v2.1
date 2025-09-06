import numpy as np
from smap_tools_python import rotate3d_vector

def test_rotate3d_vector_single():
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    v = np.array([1, 0, 0])
    out = rotate3d_vector(R, v)
    assert np.allclose(out, [0, 1, 0])

def test_rotate3d_vector_batch():
    R = np.eye(3)
    v = np.array([[1, 2, 3], [4, 5, 6]])
    out = rotate3d_vector(R, v)
    assert np.allclose(out, v)
