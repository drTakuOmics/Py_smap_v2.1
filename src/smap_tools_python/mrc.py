"""Minimal MRC file I/O helpers using the mrcfile library."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - mrcfile may be missing
    import mrcfile
except Exception:  # pragma: no cover
    mrcfile = None


def read_mrc(path: str):
    """Read an MRC file and return the data array and voxel size.

    Parameters
    ----------
    path : str
        Path to the MRC file on disk.

    Returns
    -------
    tuple
        ``(data, voxel_size)`` where ``data`` is a ``numpy.ndarray`` and
        ``voxel_size`` is a 3-tuple giving the voxel spacing in angstroms.
    """

    if mrcfile is None:  # pragma: no cover
        raise ImportError("mrcfile is required to read MRC files")

    with mrcfile.open(path, permissive=True) as mrc:
        data = np.array(mrc.data, copy=True)
        voxel = tuple(float(v) for v in mrc.voxel_size)
    return data, voxel


def write_mrc(path: str, data, voxel_size=1.0, overwrite=True):
    """Write ``data`` to ``path`` as an MRC file.

    Parameters
    ----------
    path : str
        Output filename.
    data : array_like
        Array of data to store. It will be converted to ``float32``.
    voxel_size : float or sequence of float, optional
        Voxel size in angstroms. A scalar applies to all axes.
    overwrite : bool, optional
        Whether to overwrite an existing file.
    """

    if mrcfile is None:  # pragma: no cover
        raise ImportError("mrcfile is required to write MRC files")

    arr = np.asarray(data, dtype=np.float32)
    with mrcfile.new(path, overwrite=overwrite) as mrc:
        mrc.set_data(arr)
        if np.isscalar(voxel_size):
            mrc.voxel_size = tuple([float(voxel_size)] * 3)
        else:
            mrc.voxel_size = tuple(float(v) for v in voxel_size)
