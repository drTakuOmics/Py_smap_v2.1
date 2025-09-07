import numpy as np
from smap_tools_python.phase_shift import apply_phase_shifts


def test_apply_phase_shifts_integer():
    arr = np.arange(16).reshape(4, 4)
    out = apply_phase_shifts(arr, (1, -1))
    expected = np.roll(arr, (1, -1), axis=(0, 1))
    assert np.allclose(out, expected)


def test_apply_phase_shifts_invertible():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((8, 8))
    shift = (0.3, -0.7)
    shifted = apply_phase_shifts(arr, shift)
    restored = apply_phase_shifts(shifted, (-shift[0], -shift[1]))
    assert np.allclose(restored, arr)
