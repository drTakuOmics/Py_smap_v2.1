"""Read Digital Micrograph DM3/DM4 files.

This is a lightweight translation of SMAP's ``ReadDMFile`` MATLAB helper
which extracts the image stack along with pixel size information.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np

try:  # pragma: no cover - optional dependency
    from ncempy.io import dm
except Exception:  # pragma: no cover
    dm = None  # type: ignore


def read_dm_file(path: str | Path) -> Tuple[np.ndarray, Tuple[float, float], str]:
    """Load a DM3/DM4 file using :mod:`ncempy`.

    Parameters
    ----------
    path : str or :class:`~pathlib.Path`
        Input filename.

    Returns
    -------
    tuple
        ``(data, pixel_size, units)`` where ``data`` is a ``numpy.ndarray``,
        ``pixel_size`` is a two element tuple giving the pixel spacing
        in nanometers and ``units`` is the raw unit string stored in the
        file metadata.
    """
    if dm is None:  # pragma: no cover
        raise ImportError("ncempy is required to read DM files")

    result = dm.dmReader(str(path))
    data = np.asarray(result["data"])
    px = result.get("pixelSize", (1.0, 1.0))
    units = result.get("pixelUnit", "")

    # ncempy returns pixel size in meters; convert to nanometers
    if np.isscalar(px):
        px_nm = (float(px) * 1e9, float(px) * 1e9)
    else:
        px_nm = tuple(float(v) * 1e9 for v in np.atleast_1d(px)[:2])

    return data, px_nm, str(units)
