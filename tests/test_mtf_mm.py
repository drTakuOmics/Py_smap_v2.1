import numpy as np
from smap_tools_python import mtf_mm


def test_mtf_mm_matches_matlab_example():
    params = [-0.1366, 0.0975, 0.0391, -0.0002, 1.7061, 57.4421]
    k = np.array([0.0, 0.5, 1.0])
    out = mtf_mm(params, k)
    expected = np.array([1.0, 0.77990543, 0.5000939])
    assert np.allclose(out, expected, atol=1e-6)
