import numpy as np
from smap_tools_python import ftj, iftj


def test_fft_roundtrip():
    arr = np.random.rand(4, 4)
    rec = iftj(ftj(arr))
    assert np.allclose(arr, rec, atol=1e-6)


def test_matches_matlab_fixture():
    arr = np.arange(16.0).reshape(4, 4)
    expected = np.array(
        [
            [0.0, 0.0, -8.0, 0.0],
            [0.0, 0.0, 8.0 + 8.0j, 0.0],
            [-2.0, 2.0 + 2.0j, 30.0, 2.0 - 2.0j],
            [0.0, 0.0, 8.0 - 8.0j, 0.0],
        ],
        dtype=np.complex128,
    )
    out = ftj(arr)
    assert np.allclose(out, expected)
