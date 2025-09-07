import numpy as np
from smap_tools_python.crop_patch import crop_patch_from_image


def test_crop_patch_from_image():
    arr = np.arange(100).reshape(10, 10)
    half = 2
    row_col = (2, 7)
    patch = crop_patch_from_image(arr, half, row_col)
    center = np.array(arr.shape) // 2
    shift = center - np.array(row_col)
    expected_full = np.roll(arr, shift, axis=(0, 1))
    expected = expected_full[center[0]-half:center[0]+half, center[1]-half:center[1]+half]
    assert np.allclose(patch, expected)
