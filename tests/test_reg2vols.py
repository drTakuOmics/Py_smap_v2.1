import numpy as np
from smap_tools_python import reg2vols, apply_phase_shifts


def test_reg2vols_recovers_shift():
    rng = np.random.default_rng(1)
    N = 32
    x = np.linspace(-1, 1, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    vol = np.exp(-(X**2 + Y**2 + Z**2) / 0.1)
    shift = np.array([1.5, -2.3, 0.7])
    shifted = apply_phase_shifts(vol, shift)
    reg, shifts, peak = reg2vols(shifted, vol, rtu=32)
    assert np.allclose(shifts, shift, atol=0.5)
    assert np.allclose(reg.real, vol, atol=1e-1)
