import numpy as np
from smap_tools_python import occ


def test_occ_identity_peak_at_center():
    rng = np.random.default_rng(0)
    im = rng.random((8, 8))
    out = occ(im, im)
    assert out.shape == im.shape
    center = (im.shape[0] // 2, im.shape[1] // 2)
    assert np.isclose(out[center], out.max())
