import numpy as np
from smap_tools_python import gridded_qs, measure_qd
from scipy.spatial.transform import Rotation as R


def test_gridded_qs_range_and_shape():
    mats = gridded_qs(10, 10)
    assert mats.shape[1:] == (3, 3)
    rot = R.from_matrix(mats)
    quats = rot.as_quat()
    quats = np.concatenate([quats[:, 3:4], quats[:, :3]], axis=1)
    d = measure_qd([1, 0, 0, 0], quats)
    assert np.all(d <= 10 + 1e-6)


def test_gridded_qs_zero_range():
    mats = gridded_qs(0, 5)
    assert mats.shape[0] == 1
    assert np.allclose(mats[0], np.eye(3))
