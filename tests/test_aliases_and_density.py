import numpy as np
from scipy.spatial.transform import Rotation as R

from smap_tools_python import (
    cos_mask,
    cosMask,
    radial_average_im,
    radialAverageIm,
    q2r,
    q2R,
    q_to_density,
)


def test_cosMask_alias():
    m1 = cos_mask(8, (2.0, 1.0), 1.0)
    m2 = cosMask(8, (2.0, 1.0), 1.0)
    assert np.allclose(m1, m2)


def test_radialAverageIm_alias():
    im = np.arange(9, dtype=float).reshape(3, 3)
    assert np.allclose(radial_average_im(im), radialAverageIm(im))


def test_q2R_alias():
    q = [1.0, 0.0, 0.0, 0.0]
    assert np.allclose(q2r(q), q2R(q))


def test_q_to_density_shapes_and_counts():
    mats = R.random(100).as_matrix()
    V_a, V_b = q_to_density(mats, mats, n_bins=8)
    assert V_a.shape == (8, 8, 8)
    assert V_b.shape == (8, 8, 8)
    assert V_a.sum() == 100
    assert V_b.sum() == 100
