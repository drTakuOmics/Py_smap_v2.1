import numpy as np
from smap_tools_python import make_filt, make_phase_plate


def test_make_filt_basic():
    mask = make_filt(32, (0.1, 0.2), 1)
    assert mask.shape == (32, 32)
    assert mask.max() <= 1.0 and mask.min() >= 0.0
    # center should be near unity, corner near zero
    assert mask[16, 16] > mask[0, 0]


def test_make_phase_plate_shape_and_values():
    im = np.ones((32, 32))
    pp = make_phase_plate(im)
    assert pp.shape == (32, 32)
    assert np.isclose(pp[16, 16], 1.0, atol=1e-6)
    assert np.allclose(np.abs(pp), 1.0, atol=1e-6)
