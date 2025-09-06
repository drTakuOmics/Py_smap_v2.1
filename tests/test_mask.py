import numpy as np
from src.smap_tools_python import mask_central_cross


def test_mask_central_cross_frequency():
    F = np.ones((5, 5), dtype=complex)
    out = mask_central_cross(F)
    cp = 5 // 2
    assert out[cp, cp] == 1
    assert np.all(out[cp, :cp] == 0) and np.all(out[cp, cp+1:] == 0)
    assert np.all(out[:cp, cp] == 0) and np.all(out[cp+1:, cp] == 0)
