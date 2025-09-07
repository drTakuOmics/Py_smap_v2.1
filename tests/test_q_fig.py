from pathlib import Path

import pytest

from smap_tools_python import q_fig


def test_q_fig_creates_file(tmp_path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        pytest.skip("matplotlib not installed")
    plt.figure()
    plt.plot([0, 1], [0, 1])
    out = tmp_path / 'fig.eps'
    q_fig(out)
    assert out.exists()
    # basic sanity: file should start with EPS header
    assert out.read_text().startswith('%!PS')
