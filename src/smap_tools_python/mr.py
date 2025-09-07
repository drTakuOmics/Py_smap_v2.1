import numpy as np

from .mrc import read_mrc


def mr(filename, start_slice=1, num_slices=None):
    """Read an MRC file with MATLAB-compatible orientation.

    Parameters
    ----------
    filename : str
        Path to the MRC file.
    start_slice : int, optional
        1-based index of the first slice to read. Defaults to 1.
    num_slices : int or None, optional
        Number of slices to read. ``None`` or ``float('inf')`` reads to the end.

    Returns
    -------
    tuple
        ``(data, voxel_size)`` where ``data`` has shape ``(ny, nx, nz)`` and the
        slices have been transposed to match SMAP's MATLAB ``mr`` helper.
    """
    data, voxel = read_mrc(filename)
    data = np.asarray(data)
    nz = data.shape[0]
    start = max(int(start_slice) - 1, 0)
    if num_slices is None or np.isinf(num_slices):
        end = nz
    else:
        end = min(start + int(num_slices), nz)
    selected = data[start:end]
    out = np.transpose(selected, (1, 2, 0)).copy()
    rez = float(voxel[0]) if np.ndim(voxel) else float(voxel)
    return out, rez
