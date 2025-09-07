from __future__ import annotations

"""Simplified Python port of ``smappoi_search_local``.

This module focuses on parameter parsing and basic table / rotation
handling so that other parts of the toolbox can build upon it.  The
original MATLAB routine performs an extensive local correlation search;
the current port only mirrors the initial I/O behaviour.
"""

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from .read_params_file import read_params_file
from .rotations_io import read_rotations_file


def _load_table(params: dict) -> pd.DataFrame:
    """Load particle/patch information from either a CSV table or a
    coordinate file.

    The MATLAB implementation supports multiple formats.  Here we handle
    two common cases using :mod:`pandas` and :mod:`numpy`.
    """

    if params.get("tableFile"):
        # Expect a CSV file with column headers.
        return pd.read_csv(params["tableFile"])

    if params.get("coordinateFile"):
        data = np.loadtxt(params["coordinateFile"], ndmin=2)
        # Use the first six columns which correspond to xyz and defocus
        # parameters in the MATLAB code.
        cols = ["x", "y", "z", "df1", "df2", "df3"]
        data = data[:, : len(cols)]
        return pd.DataFrame(data, columns=cols)

    return pd.DataFrame()


def smappoi_search_local(
    params_file: str | Path, jobref: int = 1
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load parameters, rotations and particle tables for a local search.

    Parameters
    ----------
    params_file
        Path to the ``.par`` file describing the search.
    jobref
        Index of the job; only used for logging in this lightweight
        implementation.

    Returns
    -------
    table, rotations
        ``table`` is the particle information as a :class:`~pandas.DataFrame`
        and ``rotations`` is a ``(3,3,N)`` ``numpy.ndarray`` of rotation
        matrices read from ``params.rotationsFile``.
    """

    params, fn_type = read_params_file(params_file)
    if fn_type != "search_local":
        raise ValueError(f"Expected function type 'search_local', got '{fn_type}'")

    rot_file = params.get("rotationsFile")
    rotations = (
        read_rotations_file(rot_file) if rot_file else np.empty((3, 3, 0))
    )

    table = _load_table(params)

    print(
        f"smappoi_search_local: loaded {len(table)} entries and "
        f"{rotations.shape[2]} rotations for job {jobref}"
    )

    return table, rotations


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m`` execution."""
    if argv is None:
        import sys

        argv = sys.argv[1:]
    if not argv:
        print("Usage: smappoi_search_local.py <paramsFile> [jobref]")
        return 1
    params_file = argv[0]
    jobref = int(argv[1]) if len(argv) > 1 else 1
    smappoi_search_local(params_file, jobref)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
