import numpy as np

from smap_tools_python import (
    cos_mask,
    variable_cos_mask,
    radial_mean,
    radial_average,
    radial_max,
    radialmeanj,
    radialmean_im,
    radial_average_im,
    radialmaxj,
)


def test_cos_mask_matches_variable_version():
    a = cos_mask(8, (2.0, 3.0), 1.0)
    b = variable_cos_mask(8, (2.0, 3.0), 1.0)
    assert np.allclose(a, b)


def test_radialmeanj_agrees_with_radial_mean():
    im = np.arange(16, dtype=float).reshape(4, 4)
    prof = radialmeanj(im)
    np.testing.assert_allclose(prof[: len(radial_mean(im))], radial_mean(im))


def test_radialmeanj_return_nd_and_3d():
    vol = np.ones((3, 3, 3), dtype=float)
    vol[1, 1, 1] = 3.0
    prof, nd = radialmeanj(vol, return_nd=True)
    assert prof[0] == 3.0
    assert nd.shape == vol.shape
    assert nd[1, 1, 1] == 3.0


def test_radial_average_im_alias():
    im = np.arange(16, dtype=float).reshape(4, 4)
    np.testing.assert_allclose(radial_average_im(im), radial_average(im))


def test_radialmean_im_alias():
    im = np.arange(16, dtype=float).reshape(4, 4)
    np.testing.assert_allclose(radialmean_im(im), radial_mean(im))


def test_radialmaxj_alias():
    im = np.arange(16, dtype=float).reshape(4, 4)
    np.testing.assert_allclose(radialmaxj(im), radial_max(im))
