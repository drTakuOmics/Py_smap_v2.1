import numpy as np

from smap_tools_python import ep2sp, def_consts


def test_ep2sp_center_value():
    vol = np.zeros((4, 4, 4), float)
    vol[2, 2, 2] = 2.0
    sp, a = ep2sp(vol, 1.0, 1.0)
    assert sp.shape == (4, 4, 4)
    consts = def_consts()
    scale = consts["IC"] * (10.0 / 1e10) / (2 * consts["k"])
    assert np.isclose(sp[2, 2, 2], 2.0 * scale)
    assert sp.dtype == np.float32
    assert np.isclose(a, 10.0)
