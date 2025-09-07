import numpy as np
from smap_tools_python import get_pref, mr, write_mrc, resize_F


def test_get_pref_basic():
    prefs = ["mapsDir:/tmp/maps", "datasetsDir:/tmp/data"]
    assert get_pref(prefs, "mapsDir") == "/tmp/maps"
    all_prefs = get_pref(prefs, "all")
    assert all_prefs["datasetsDir"] == "/tmp/data"


def test_mr_reads_with_transpose(tmp_path):
    data = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)
    path = tmp_path / "test.mrc"
    write_mrc(path, data, voxel_size=1.5)
    out, rez = mr(path, start_slice=2, num_slices=2)
    assert rez == 1.5
    expected = np.transpose(data[1:3], (1, 2, 0))
    assert np.array_equal(out, expected)


def test_resize_f_runs():
    arr = np.ones((4, 4))
    out = resize_F(arr, 0.5, method="newSize")
    assert out.shape == (2, 2)
