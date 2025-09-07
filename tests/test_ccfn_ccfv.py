import numpy as np
from smap_tools_python import ccfn, ccfv


def test_ccfn_peak():
    rng = np.random.default_rng(0)
    image = rng.normal(size=(8, 8))
    templates = np.stack([image.copy(), rng.normal(size=(8, 8))], axis=-1)
    cc, peaks = ccfn(image, templates)
    assert cc.shape == (8, 8, 2)
    assert peaks[0] > peaks[1]
    max_pos = np.unravel_index(np.argmax(cc[:, :, 0]), (8, 8))
    assert max_pos == (4, 4)


def test_ccfv_peak():
    rng = np.random.default_rng(0)
    image = rng.normal(size=(8, 8))
    templates = np.stack([image.copy(), rng.normal(size=(8, 8))], axis=-1)
    cc, peaks = ccfv(image, templates)
    assert cc.shape == (8, 8, 2)
    assert peaks[0] > peaks[1]
    max_pos = np.unravel_index(np.argmax(cc[:, :, 0]), (8, 8))
    assert max_pos == (4, 4)
