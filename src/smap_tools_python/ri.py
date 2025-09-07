"""Image reading utilities translating MATLAB's ``ri`` and ``tr`` helpers."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .mrc import read_mrc

try:  # pragma: no cover - optional dependency
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None


def tr(filename: str | Path):
    """Read a TIFF file returning an array shaped ``(H, W, N)``.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the TIFF file.  If no extension is provided ``.tif`` is
        appended automatically.

    Returns
    -------
    numpy.ndarray
        Image data with the frame axis last, matching the MATLAB ``tr``
        helper.  Single-frame images retain a third axis of length one.
    """

    if tifffile is None:  # pragma: no cover
        raise ImportError("tifffile is required to read TIFF files")

    path = Path(filename)
    if path.suffix == "":
        path = path.with_suffix(".tif")

    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    else:
        arr = np.moveaxis(arr, 0, -1)
    return arr


def ri(filename: str | Path):
    """Read an image or volume in TIFF or MRC format.

    Parameters
    ----------
    filename : str or pathlib.Path
        Input file path.

    Returns
    -------
    tuple
        ``(data, info)`` where ``info`` is a dictionary that may contain
        ``"voxel_size"`` for MRC files.  For TIFF files the dictionary is
        empty.
    """

    path = Path(filename)
    ext = path.suffix.lower()

    if ext in (".tif", ".tiff", ""):
        data = tr(path)
        return data, {}
    if ext == ".mrc":
        data, voxel = read_mrc(str(path))
        return data, {"voxel_size": voxel}
    if ext == ".dm4":  # pragma: no cover - not yet implemented
        raise NotImplementedError("DM4 reading not implemented")
    raise ValueError(f"Unknown file type: {ext}")
