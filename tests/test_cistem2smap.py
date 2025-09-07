import numpy as np
from scipy.spatial.transform import Rotation as R
from smap_tools_python.cistem2smap import cistem2smap
from smap_tools_python.rotate import normalize_rotation_matrices


def test_cistem2smap_matches_rotation():
    ea = np.array([[10.0, 20.0, 30.0], [45.0, 60.0, 90.0]])
    out = cistem2smap(ea)
    ang = np.deg2rad(ea[:, ::-1])
    expected = R.from_euler("ZYZ", ang).as_matrix().transpose(0, 2, 1)
    expected = normalize_rotation_matrices(expected).transpose(1, 2, 0)
    assert out.shape == (3, 3, ea.shape[0])
    assert np.allclose(out, expected)
