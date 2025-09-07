import numpy as np
from smap_tools_python import ccff, ccff_bak_041423


def test_ccff_bak_equivalence():
    rng = np.random.default_rng(0)
    image = rng.normal(size=(8, 8))
    templates = np.stack([image.copy(), rng.normal(size=(8, 8))], axis=-1)
    cc1, peaks1 = ccff(image, templates, mode="noFilt")
    cc2, peaks2 = ccff_bak_041423(image, templates, mode="noFilt")
    assert np.allclose(cc1, cc2)
    assert np.allclose(peaks1, peaks2)
