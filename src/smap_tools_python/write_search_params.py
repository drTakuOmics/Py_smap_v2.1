from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any


def write_search_params(path: str | Path, params: Mapping[str, Any]) -> None:
    """Write search parameters to ``path``.

    The MATLAB ``writeSearchParams`` helper persisted job submission
    parameters for later inspection. This Python port simply writes the
    key-value pairs to a text file using ``key=value`` lines sorted by
    key. Values are converted to strings with ``str``.
    """
    path = Path(path)
    with path.open("w") as f:
        for key in sorted(params):
            f.write(f"{key}={params[key]}\n")


# MATLAB-style alias
writeSearchParams = write_search_params
