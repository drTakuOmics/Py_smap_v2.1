import numpy as np
from smap_tools_python import sum_frames


def test_sum_frames_first_axis():
    stack = np.arange(5)[:, None, None] * np.ones((5, 4, 4))
    summed, n = sum_frames(stack, target_exposure=7, exposure_per_frame=2, axis=0)
    assert n == 4
    assert np.allclose(summed, stack[:4].sum(axis=0))


def test_sum_frames_last_axis():
    stack = np.arange(5)[None, None, :] * np.ones((4, 4, 5))
    summed, n = sum_frames(stack, target_exposure=6, exposure_per_frame=2, axis=-1)
    assert n == 3
    assert np.allclose(summed, stack[:, :, :3].sum(axis=2))
