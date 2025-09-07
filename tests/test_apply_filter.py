import numpy as np
from smap_tools_python import apply_filter


def test_apply_filter_identity():
    im = np.random.rand(32, 32)
    filt = np.ones_like(im)
    out = apply_filter(im, filt, norm=False)
    assert np.allclose(out, im)
