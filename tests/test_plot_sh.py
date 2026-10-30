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


def test_plot_sh_separate_null_population():
    """Regression for the reviewed issue: 3 scores with Nsamples=100."""
    v = np.array([0.0, 1.0, 2.0])
    xs, ys, YS, thr = plot_sh(v, Nsamples=100)
    assert ys[0] == 2                       # count(v > 0)
    assert ys[-1] == int(np.sum(v > xs[-1])) == 1
    assert np.array_equal(ys[::997], np.array([np.sum(v > x) for x in xs[::997]]))
