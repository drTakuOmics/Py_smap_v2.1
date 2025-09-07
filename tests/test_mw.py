import numpy as np
import pytest
from smap_tools_python import mw, read_mrc

pytest.importorskip("mrcfile")


def test_mw_roundtrip(tmp_path):
    vol = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    out = tmp_path / "test.mrc"
    mw(vol, out, 1.5)
    data, voxel = read_mrc(out)
    assert data.shape == (4, 3, 2)
    assert np.allclose(data, vol.swapaxes(0, 1))
    assert voxel == (1.5, 1.5, 1.5)
