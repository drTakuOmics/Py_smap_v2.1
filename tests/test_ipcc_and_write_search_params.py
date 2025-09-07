import numpy as np
from pathlib import Path

from smap_tools_python import ftj, ipcc, write_search_params


def test_ipcc_rotation_detection():
    n = 64
    im = np.zeros((n, n))
    im[40, 10] = 1  # asymmetric feature
    rot = np.rot90(im)
    im_f = np.abs(ftj(im))
    rot_f = np.abs(ftj(rot))
    cc = ipcc(im_f, rot_f)
    peak = np.unravel_index(np.argmax(cc), cc.shape)
    angle = peak[1] * 360 / cc.shape[1]
    if angle > 180:
        angle = 360 - angle
    assert abs(angle - 90) < 1


def test_write_search_params(tmp_path: Path):
    params = {"b": 2, "a": 1}
    out = tmp_path / "params.txt"
    write_search_params(out, params)
    assert out.read_text().splitlines() == ["a=1", "b=2"]
