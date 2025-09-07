import numpy as np
import pytest

from smap_tools_python import read_mrc, write_mrc

pytest.importorskip("mrcfile")


def test_mrc_roundtrip(tmp_path):
    arr = np.random.rand(4, 4).astype(np.float32)
    path = tmp_path / "t.mrc"
    write_mrc(path, arr, voxel_size=1.5)
    out, vox = read_mrc(path)
    assert np.allclose(out, arr)
    assert np.allclose(vox, (1.5, 1.5, 1.5))
