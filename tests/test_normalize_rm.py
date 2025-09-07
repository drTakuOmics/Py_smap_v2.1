import numpy as np
from smap_tools_python import normalize_rm


def test_normalize_rm_single():
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    R_noisy = R + 0.01 * np.eye(3)
    Rn = normalize_rm(R_noisy)
    assert np.allclose(Rn @ Rn.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(Rn), 1.0, atol=1e-6)


def test_normalize_rm_stack():
    R = np.eye(3)
    stack = np.stack([R + 0.01 * np.eye(3), R - 0.02 * np.eye(3)], axis=2)
    Rn = normalize_rm(stack)
    assert Rn.shape == stack.shape
    for i in range(Rn.shape[2]):
        Ri = Rn[:, :, i]
        assert np.allclose(Ri @ Ri.T, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(Ri), 1.0, atol=1e-6)
