import numpy as np
from smap_tools_python.radial import radial_mean, radial_average, radial_max


def test_radial_mean_and_average():
    im = np.zeros((5, 5), float)
    y, x = np.indices(im.shape)
    r = np.round(np.sqrt((x - 2) ** 2 + (y - 2) ** 2)).astype(int)
    im = r.astype(float)
    prof = radial_mean(im)
    assert np.allclose(prof[: r.max() + 1], np.arange(r.max() + 1))
    avg = radial_average(im)
    assert np.allclose(avg, im)


def test_radial_max():
    im = np.zeros((7, 7))
    im[3, 3] = 1
    im[3, 5] = 2  # radius ~2
    m = radial_max(im)
    assert m[0] == 1
    assert m[2] == 2
