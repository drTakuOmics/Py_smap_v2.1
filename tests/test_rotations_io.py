import numpy as np
from smap_tools_python import write_rotations_file, read_rotations_file


def test_rotation_roundtrip(tmp_path):
    R = np.stack([
        np.eye(3),
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float),
    ], axis=2)
    fn = tmp_path / "rots.txt"
    write_rotations_file(R, fn)
    R2 = read_rotations_file(fn)
    assert np.allclose(R2, R)
