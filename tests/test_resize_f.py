import numpy as np
from smap_tools_python import resize_F


def test_resize_f_shapes():
    arr = np.random.rand(8, 8)
    out_new = resize_F(arr, 1.5, method='newSize')
    assert out_new.shape == (12, 12)
    out_fixed = resize_F(arr, 1.5, method='fixedSize')
    assert out_fixed.shape == arr.shape


def test_resize_f_round_trip():
    arr = np.random.rand(8, 8)
    down = resize_F(arr, 0.5, method='newSize')
    up = resize_F(down, 2, method='newSize')
    assert up.shape == arr.shape
    assert np.isfinite(up).all()

