import numpy as np
from smap_tools_python import write_dat, read_dat_file


def test_write_dat(tmp_path):
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    fn = tmp_path / "data.dat"
    write_dat(arr, fn)
    raw = np.fromfile(fn, dtype=np.float32)
    assert np.allclose(raw, arr.ravel())
    hdr = fn.with_suffix(".hdr").read_text().strip()
    assert hdr == "\t".join(["2", "3", "4"])


def test_read_dat_file(tmp_path):
    values = np.array([1, 10, 100, 2, 20, 200], dtype=np.float64)
    fn = tmp_path / "triples.dat"
    values.tofile(fn)
    ai, al, av = read_dat_file(fn)
    assert np.allclose(ai, [1, 2])
    assert np.allclose(al, [10, 20])
    assert np.allclose(av, [100, 200])
