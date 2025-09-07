import numpy as np
from smap_tools_python import backproject


def test_backproject_identity_slice():
    patch = np.array([[1, 2], [3, 4]], float)
    patches = patch[:, :, None]
    R = np.eye(3)[None, ...]
    vol = backproject(patches, R, pad_size=4)
    expected = np.array(
        [
            [2.5, 2.5, 2.5, 2.5],
            [2.5, 1.0, 2.0, 2.5],
            [2.5, 3.0, 4.0, 2.5],
            [2.5, 2.5, 2.5, 2.5],
        ]
    )
    assert np.allclose(vol[:, :, 2], expected)
    assert np.allclose(vol[:, :, :2], 0)
    assert np.allclose(vol[:, :, 3], 0)
