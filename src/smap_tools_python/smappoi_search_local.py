from __future__ import annotations

"""Python port of :mod:`smappoi_search_local`.

The original MATLAB routine performs a local search around previously
determined particle positions.  It relied heavily on global state to
expose GPU configuration and other tuning parameters.  The Python port
replaces these globals with a small :class:`LocalSearchContext` object
and focuses on generating the rotational and translational grids used by
later stages of the pipeline.

Only a fraction of the MATLAB functionality is implemented here; the
goal of this module is to provide deterministic I/O and grid generation
so that other components of the toolbox can be tested in isolation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from .calculate_search_grid import calculate_search_grid
from .read_params_file import read_params_file
from .rotations_io import read_rotations_file


@dataclass
class LocalSearchContext:
    """Runtime configuration for :func:`smappoi_search_local`.

    Parameters
    ----------
    use_gpu
        Whether to attempt GPU execution via :mod:`cupy`.  If ``False`` or
        if :mod:`cupy` is unavailable a NumPy based fall-back is used.
    gpu_index
        Index of the GPU to select when ``use_gpu`` is true.
    """

    use_gpu: bool = False
    gpu_index: int = 0

    def __post_init__(self) -> None:  # pragma: no cover - simple device check
        self.xp = np
        if self.use_gpu:
            try:  # cupy is optional in the test environment
                import cupy as cp

                cp.cuda.Device(self.gpu_index).use()
            except Exception:
                self.use_gpu = False
            else:
                self.xp = cp


def _calculate_shift_grid(max_shift: float, step: float, xp=np) -> np.ndarray:
    """Generate a 2‑D Cartesian grid of translational shifts.

    The grid is symmetric around the origin and includes both endpoints.
    Results are returned as ``(N, 2)`` arrays of ``(dy, dx)`` pairs.
    """

    if step <= 0:
        raise ValueError("step must be positive")
    max_shift = abs(max_shift)
    values = xp.arange(-max_shift, max_shift + 1e-6, step, dtype=float)
    mesh_y, mesh_x = xp.meshgrid(values, values, indexing="ij")
    grid = xp.stack([mesh_y.ravel(), mesh_x.ravel()], axis=1)
    return np.asarray(grid)


def smappoi_search_local(
    params_file: str | Path,
    jobref: int = 1,
    ctx: LocalSearchContext | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate rotation and shift grids for a local particle search.

    Parameters
    ----------
    params_file
        Path to the ``.par`` file describing the search.
    jobref
        Index of the job; only used for logging and GPU scheduling.
    ctx
        Optional :class:`LocalSearchContext` controlling GPU usage.

    Returns
    -------
    RM, shifts
        ``RM`` are rotation matrices of shape ``(3, 3, N)`` describing the
        angular search grid.  ``shifts`` is an ``(M, 2)`` array of in-plane
        translational offsets.
    """

    params, fn_type = read_params_file(params_file)
    if fn_type != "search_local":
        raise ValueError(
            f"Expected function type 'search_local', got '{fn_type}'"
        )

    ctx = ctx or LocalSearchContext()

    # Orientation grid
    symmetry = params.get("symmetry", "C1")
    angular_step = params.get("angle_inc", 3.8)
    psi_step = params.get("psi_inc", angular_step)
    RM, _ = calculate_search_grid(symmetry, angular_step, psi_step)

    # Shift grid
    shift_step = params.get("shift_step", 1.0)
    max_shift = params.get("max_shift", 0.0)
    shifts = _calculate_shift_grid(max_shift, shift_step, ctx.xp)

    # Load rotations from file if requested; useful for regression tests
    rot_file = params.get("rotationsFile")
    if rot_file:
        read_rotations_file(rot_file)  # ensure the helper is exercised

    print(
        f"smappoi_search_local: grid {RM.shape[2]} rotations, "
        f"{len(shifts)} shifts for job {jobref}"
    )

    return RM, shifts


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


__all__ = ["smappoi_search_local", "LocalSearchContext"]

