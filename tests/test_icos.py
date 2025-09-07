import numpy as np
from scipy.spatial.transform import Rotation

from smap_tools_python import icos, get_icos


def test_icos_shapes_and_radius():
    xyz_sub, xyz_rnap = icos()
    assert xyz_sub.shape == (60, 3)
    assert xyz_rnap.shape == (60, 3)

    radii = np.linalg.norm(xyz_sub, axis=1)
    assert np.allclose(radii, radii[0])
    assert np.isclose(radii[0], 1.0, atol=1e-6)


def test_get_icos_identity_matches_group():
    q_out, xyz_sub, xyz_rnap = get_icos([0, 0, 0, 1])
    assert q_out.shape == (60, 4)

    ops = Rotation.create_group("I")
    np.testing.assert_allclose(
        Rotation.from_quat(q_out).as_matrix(), ops.as_matrix()
    )

    xyz_sub_base, xyz_rnap_base = icos()
    np.testing.assert_allclose(xyz_sub, xyz_sub_base)
    np.testing.assert_allclose(xyz_rnap, xyz_rnap_base)

