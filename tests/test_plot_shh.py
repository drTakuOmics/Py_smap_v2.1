import numpy as np
import pytest
from scipy.special import erfcinv
from smap_tools_python import plot_shh


def test_plot_shh_threshold_and_plot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        pytest.skip("matplotlib not installed")
    bins = np.array([0, 1, 2, 3])
    N = np.array([[5], [3], [2], [0]])
    fig, ax = plt.subplots()
    thr = plot_shh(bins, N, nfp=1, ax=ax)
    expected = np.sqrt(2) * erfcinv(1 * 2 / N.sum())
    assert np.isclose(thr, expected)
    assert len(ax.lines) >= 2
