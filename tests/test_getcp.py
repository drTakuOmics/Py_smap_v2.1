import numpy as np
from smap_tools_python import get_center_pixel, getcp


def test_get_center_pixel_from_array():
    arr = np.zeros((5, 7, 9))
    assert get_center_pixel(arr) == (2, 3, 4)


def test_getcp_from_shape_tuple():
    assert getcp((6, 6)) == (3, 3)
