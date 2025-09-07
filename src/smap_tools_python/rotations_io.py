import numpy as np
from pathlib import Path


def write_rotations_file(R, filename="testR.txt"):
    """Write rotation matrices to an ASCII file.

    Parameters
    ----------
    R : array_like, shape (3, 3, N)
        Rotation matrices to write.
    filename : str or path-like, optional
        Destination file; defaults to ``"testR.txt"``.
    """
    arr = np.asarray(R, dtype=float)
    if arr.shape[0:2] != (3, 3):
        raise ValueError("Input must have shape (3,3,N)")
    mats = arr.reshape(3, 3, -1)
    path = Path(filename)
    with path.open("w") as fh:
        for i in range(mats.shape[2]):
            Rf = mats[:, :, i].T
            for row in Rf:
                fh.write(f"{i + 1:7d}\t{row[0]:5.4f}\t{row[1]:5.4f}\t{row[2]:5.4f}\n")


def read_rotations_file(filename):
    """Read rotation matrices from an ASCII file produced by
    :func:`write_rotations_file`.
    """
    data = np.loadtxt(filename)
    rows = data[:, 1:]
    n = rows.shape[0] // 3
    mats_t = rows.reshape(n, 3, 3)
    mats = np.transpose(mats_t, (0, 2, 1))
    return mats.transpose(1, 2, 0)
