import numpy as np
from smap_tools_python import backproject


def test_backproject_identity_slice():
    patch = np.array([[1, 2], [3, 4]], float)
    patches = patch[:, :, None]
    R = np.eye(3)[None, ...]
    vol, weights, other = backproject(patches, R, pad_size=4)

    reconstructed = np.divide(vol, weights, out=np.zeros_like(vol), where=weights > 0)
    expected = np.zeros((4, 4))
    expected[1:3, 1:3] = patch

    assert np.allclose(reconstructed[:, :, 2], expected)
    assert np.allclose(reconstructed[:, :, :2], 0)
    assert np.allclose(reconstructed[:, :, 3], 0)
    assert np.allclose(other, 0)
