from __future__ import annotations

"""Simplified Python port of ``smappoi_search_global``.

This module provides a lightweight translation of the MATLAB
``smappoi_search_global`` routine.  The original implementation
contains thousands of lines handling GPU setup, FFTs, and search
optimisation.  The current port focuses on the initial parameter
parsing and search grid calculation so that other Python components
can build upon it.

Notes
-----
This file is an early-stage port.  Many features of the MATLAB
version are intentionally omitted.  Further work is required to
implement the complete global search pipeline.
"""

from pathlib import Path
import sys
from typing import Sequence

from .read_params_file import read_params_file
from .calculate_search_grid import calculate_search_grid


def smappoi_search_global(params_file: str | Path, jobref: int = 1) -> None:
    """Run a minimal global search using Python utilities.

    Parameters
    ----------
    params_file
        Path to the ``.par`` file describing the search.
    jobref
        Index of the job for GPU scheduling.  Only used for logging
        in this lightweight implementation.
    """
    params, fn_type = read_params_file(params_file)
    if fn_type != "search_global":
        raise ValueError(
            f"Expected function type 'search_global', got '{fn_type}'"
        )

    # Calculate the search grid to validate parameters.  The result is
    # currently discarded but exercises the translated helper
    # functions.  ``calculate_search_grid`` expects explicit symmetry
    # and sampling information so we pull a minimal set of values from
    # ``params`` with conservative defaults that mirror the behaviour
    # of the MATLAB version.
    symmetry = params.get("symmetry", "C1")
    angular_step = params.get("angle_inc", 3.8)
    psi_step = params.get("psi_inc", angular_step)
    calculate_search_grid(symmetry, angular_step, psi_step)

    print(
        f"smappoi_search_global: loaded {len(params)} parameters for job {jobref}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m`` execution."""
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print("Usage: smappoi_search_global.py <paramsFile> [jobref]")
        return 1
    params_file = argv[0]
    jobref = int(argv[1]) if len(argv) > 1 else 1
    smappoi_search_global(params_file, jobref)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
