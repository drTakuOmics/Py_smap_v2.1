import numpy as np
from numpy.fft import fftn, ifftn
from scipy.ndimage import fourier_shift

from smap_tools_python.motion_corr import motion_corr


def test_motion_corr_aligns_frames():
    base = np.zeros((32, 32))
    base[10:22, 8:20] = 1
    shifts = [(0, 0), (1, -2), (-3, 2), (2, 1)]
    frames = []
    for sh in shifts:
        shifted = ifftn(fourier_shift(fftn(base), sh)).real
        frames.append(shifted)
    movie = np.stack(frames, axis=0)
    aligned, est = motion_corr(movie, ref_index=0, axis=0)
    # after correction each frame should match the reference pattern
    assert np.allclose(aligned, base, atol=1e-3)
    assert np.allclose(est, -np.array(shifts), atol=0.5)
