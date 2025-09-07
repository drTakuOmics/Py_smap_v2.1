import numpy as np
from smap_tools_python import pad_for_fft


def test_pad_for_fft_crops_to_36():
    arr = np.ones((40, 40))
    out = pad_for_fft(arr)
    assert out.shape == (36, 36)
