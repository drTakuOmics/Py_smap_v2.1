import numpy as np
from smap_tools_python import crop_or_pad, resize_for_fft

def test_crop_or_pad_crop():
    arr = np.arange(16).reshape(4, 4)
    out = crop_or_pad(arr, (2, 2))
    expected = np.array([[5, 6], [9, 10]])
    assert np.array_equal(out, expected)

def test_crop_or_pad_pad():
    arr = np.array([[1, 2], [3, 4]])
    out = crop_or_pad(arr, (4, 4), pad_value=1)
    expected = np.array([[1, 1, 1, 1],
                         [1, 1, 2, 1],
                         [1, 3, 4, 1],
                         [1, 1, 1, 1]])
    assert np.array_equal(out, expected)

def test_resize_for_fft_pad_and_crop():
    arr_pad = np.zeros((5, 5))
    out_pad = resize_for_fft(arr_pad, mode='pad')
    assert out_pad.shape == (6, 6)
    arr_crop = np.zeros((7, 7))
    out_crop = resize_for_fft(arr_crop, mode='crop')
    assert out_crop.shape == (6, 6)
