import numpy as np
from src.smap_tools_python import ftj, iftj


def test_fft_roundtrip():
    arr = np.random.rand(4, 4)
    rec = iftj(ftj(arr))
    assert np.allclose(arr, rec, atol=1e-6)
