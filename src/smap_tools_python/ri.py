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


def tw(array, filename: str | Path, bps: int = 32):
    """Write an array to a TIFF file with configurable bit depth.

    Parameters
    ----------
    array : array_like
        Image data shaped ``(H, W)`` or ``(H, W, N)`` where the last dimension
        enumerates frames.
    filename : str or pathlib.Path
        Destination path. A ``.tif`` extension is appended if missing.
    bps : int, optional
        Bits per sample. Supported values are ``16`` for unsigned integers and
        ``32`` for IEEE floating point. Defaults to ``32``.
    """

    if tifffile is None:  # pragma: no cover
        raise ImportError("tifffile is required to write TIFF files")

    path = Path(filename)
    if path.suffix == "":
        path = path.with_suffix(".tif")

    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]

    if bps == 16:
        arr = arr.astype(np.uint16, copy=False)
    elif bps == 32:
        arr = arr.astype(np.float32, copy=False)
    else:
        raise ValueError("Only 16 or 32 bits per sample are supported")

    tifffile.imwrite(str(path), np.moveaxis(arr, -1, 0))


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
