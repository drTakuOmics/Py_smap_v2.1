import numpy as np
from src.smap_tools_python import mask_central_cross
from src.smap_tools_python import mask_volume









def test_mask_central_cross_frequency():
    F = np.ones((5, 5), dtype=complex)
    out = mask_central_cross(F)
    cp = 5 // 2
    assert out[cp, cp] == 1
    assert np.all(out[cp, :cp] == 0) and np.all(out[cp, cp+1:] == 0)
    assert np.all(out[:cp, cp] == 0) and np.all(out[cp+1:, cp] == 0)








def test_mask_volume_mask_mode():
    vol = np.zeros((5, 5, 5), float)
    vol[2, 2, 2] = 1
    out, mask, D = mask_volume(vol, (1, 1))
    assert out.shape == (5, 5, 5)
    assert mask.shape == (5, 5, 5)
    assert D.shape == (5, 5, 5)


def test_mask_volume_shell_mode():
    vol = np.zeros((5, 5, 5), float)
    vol[2, 2, 2] = 1
    out, mask, D = mask_volume(vol, (1, 1), mode="shell")



    assert mask.sum() > 0