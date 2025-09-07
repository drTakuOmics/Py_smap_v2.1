import numpy as np
from smap_tools_python import q2r
from smap_tools_python.quaternion import Quaternion


def test_q2r_matches_quaternion_class():
    q = np.random.randn(4)
    R1 = q2r(q)
    R2 = Quaternion(*q).to_rotation_matrix()
    assert np.allclose(R1, R2)
