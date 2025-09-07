import numpy as np
from .mrc import write_mrc


def mw(volume, filename, voxel_size):
    """Write a volume to an MRC file with axis swap.

    This mirrors MATLAB's ``mw`` helper which transposes the first two axes
    before delegating to ``WriteMRC``.  The orientation swap compensates for
    the convention used by the original reader and ensures that a read/write
    round‑trip yields an identical volume.
    """

    arr = np.asarray(volume)
    if arr.ndim != 3:
        raise ValueError("volume must be a 3‑D array")
    transposed = arr.swapaxes(0, 1)
    write_mrc(filename, transposed, voxel_size)
