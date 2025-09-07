import numpy as np
from smap_tools_python import max_interp_f, apply_phase_shifts


def test_max_interp_f_centering():
    n = 64
    arr = np.zeros((n, n))
    cy = cx = n // 2
    peak = (cy + 3, cx - 2)
    arr[peak] = 1.0
    shift, val = max_interp_f(arr)
    shifted = apply_phase_shifts(arr, shift)
    ny, nx = np.unravel_index(np.argmax(shifted), shifted.shape)
    assert (ny, nx) == (cy, cx)
    assert np.isclose(val, 1.0)
