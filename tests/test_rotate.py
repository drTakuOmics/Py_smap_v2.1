import numpy as np
from smap_tools_python import (
    rotate3d_vector,
    rotate2d_matrix,
    rotate3d_matrix,
    rot90j,
    normalize_rotation_matrices,
)

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


def test_rotate3d_vector_column():
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    v = np.array([[1, 0], [0, 1], [0, 0]])
    out = rotate3d_vector(R, v)
    expected = R @ v
    assert np.allclose(out, expected)


def test_rot90j_matches_numpy():
    arr = np.arange(9).reshape(3, 3)
    assert np.array_equal(rot90j(arr, 1), np.rot90(arr, 1))


def test_rot90j_even_shift():
    arr = np.arange(16).reshape(4, 4)
    out = rot90j(arr, 1)
    expected = np.roll(np.rot90(arr, 1), (1, 0), axis=(0, 1))
    assert np.array_equal(out, expected)


def test_rotate2d_matrix_identity():
    img = np.arange(9).reshape(3, 3)
    R = np.eye(2)
    assert np.array_equal(rotate2d_matrix(img, R), img)


def test_rotate2d_matrix_90():
    img = np.arange(9).reshape(3, 3)
    R = np.array([[0, -1], [1, 0]])
    out = rotate2d_matrix(img, R)
    assert np.array_equal(out, rot90j(img, 1))


def test_rotate3d_matrix_identity():
    vol = np.arange(27).reshape(3, 3, 3)
    R = np.eye(3)
    assert np.array_equal(rotate3d_matrix(vol, R), vol)


def test_rotate3d_matrix_z90():
    vol = np.zeros((3, 3, 3))
    vol[1, 0, 1] = 1
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    out = rotate3d_matrix(vol, R)
    expected = np.zeros_like(vol)
    expected[1, 1, 1] = 1
    assert np.array_equal(out, expected)


def test_normalize_rotation_matrices():
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    R_noisy = R + 0.01 * np.eye(3)
    Rn = normalize_rotation_matrices(R_noisy)
    assert np.allclose(Rn @ Rn.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(Rn), 1.0, atol=1e-6)

