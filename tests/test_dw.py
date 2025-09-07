import numpy as np
from smap_tools_python import dw


def test_dw_roundtrip(tmp_path):
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    fn = tmp_path / "out.dat"
    dw(arr, fn)
    data = np.fromfile(fn, dtype=np.float32).reshape(arr.shape)
    assert np.allclose(data, arr)
    hdr = (tmp_path / "out.hdr").read_text().strip()
    assert hdr == "3\t4"
