import os
from pathlib import Path


def check_base_dir(path: str = "") -> str:
    """Normalize paths relative to a configurable SMAP base directory.

    Parameters
    ----------
    path : str, optional
        Path to normalize. If empty, just return the base directory.

    Environment variables
    ---------------------
    SMAP_BASE_DIR : str
        Desired base directory prefix (default ``"/"``).
    SMAP_BASE_DIR_ACTUAL : str
        Alternate prefix that should be replaced with ``SMAP_BASE_DIR`` when
        present in ``path``.
    """
    base = os.environ.get("SMAP_BASE_DIR", "/")
    actual = os.environ.get("SMAP_BASE_DIR_ACTUAL", base)
    if not path:
        return base
    path = str(path)
    if path.startswith(actual):
        return path.replace(actual, base, 1)
    if Path(path).is_absolute():
        return path
    return str(Path(base) / path)
