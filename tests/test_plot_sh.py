import numpy as np
from scipy.special import erfcinv
from smap_tools_python.plot_sh import plot_sh


def test_plot_sh_basic():
    rng = np.random.default_rng(0)
    data = rng.normal(size=1000)
    xs, ys, YS, thr = plot_sh(data)
    assert xs.shape == ys.shape == YS.shape
    assert np.all(np.diff(ys) <= 0)
    expected_thr = np.sqrt(2) * erfcinv(2 / len(data))
    assert np.isclose(thr, expected_thr)
