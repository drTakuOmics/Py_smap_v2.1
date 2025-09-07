import numpy as np

from smap_tools_python.mw import mw
from smap_tools_python.register_multiple_fragments import register_multiple_fragments


def test_register_multiple_fragments_aligns_shifts(tmp_path):
    edge = 16
    ref = np.zeros((edge, edge, edge), dtype=float)
    ref[8, 8, 8] = 1.0
    frag = np.zeros_like(ref)
    frag[9, 7, 8] = 1.0  # shifted by (-1, +1, 0)

    ref_path = tmp_path / "ref.mrc"
    frag_path = tmp_path / "frag.mrc"
    mw(ref, ref_path.as_posix(), 1.0)
    mw(frag, frag_path.as_posix(), 1.0)

    out, shifts = register_multiple_fragments([ref_path.as_posix(), frag_path.as_posix()])
    assert np.allclose(shifts[0], [1, 0, -1], atol=1e-3)
    assert out[8, 8, 8] > 0.9

