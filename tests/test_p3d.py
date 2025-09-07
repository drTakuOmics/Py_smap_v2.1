import numpy as np
import pytest
from smap_tools_python import p3d


def test_p3d_returns_axis():
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        pytest.skip("matplotlib not installed")
    pts = np.random.rand(10, 3)
    ax = p3d(pts)
    assert ax.has_data()
