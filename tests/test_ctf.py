import numpy as np
from smap_tools_python import ctf


def test_ctf_zero_defocus():
    params = {
        "Cs": 0.0,
        "Cc": 0.0,
        "V_acc": 300e3,
        "deltaE": 0.0,
        "a_i": 0.0,
        "aPerPix": 1.0,
        "F_abs": 0.0,
    }
    out = ctf(np.array([[0.0, 0.0, 0.0]]), 4, params)
    assert out.shape == (4, 4)
    center = out.shape[0] // 2
    val = out[center, center]
    assert np.isclose(val.real, -1.0)
    assert np.isclose(val.imag, 0.0)
