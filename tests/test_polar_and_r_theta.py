import numpy as np
from smap_tools_python import polar_image, r_theta


def _radial_image(n):
    y, x = np.indices((n, n))
    c = (n - 1) / 2.0
    return np.hypot(x - c, y - c)


def test_polar_image_shape_and_monotonic():
    img = _radial_image(64)
    polar, _ = polar_image(img, 32, 90, center=((64 - 1) / 2, (64 - 1) / 2), shape="valid")
    assert polar.shape == (32, 90)
    assert np.all(np.diff(polar[:, 0]) >= 0)


def test_r_theta_statistics():
    rng = np.random.default_rng(0)
    img = rng.standard_normal((64, 64))
    rt = r_theta(img)
    assert rt.shape[0] == 360
    # r_theta normalizes with nm
    assert abs(rt.mean()) < 1e-6
    assert abs(rt.std(ddof=1) - 1) < 1e-6
