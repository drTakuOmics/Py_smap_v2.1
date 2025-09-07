import numpy as np
from pathlib import Path

from smap_tools_python import smappoi_search_local


def test_smappoi_search_local_grid_and_shifts(tmp_path):
    fixture_dir = Path("tests/fixtures")

    params_file = tmp_path / "params.par"
    params_file.write_text(
        f"""
function search_local
symmetry C1
angle_inc 90
psi_inc 180
shift_step 2
max_shift 4
"""
    )

    grid, shifts = smappoi_search_local(params_file)

    expected_grid = (
        np.loadtxt(fixture_dir / "local_grid.txt")
        .reshape(-1, 3, 3)
        .transpose(1, 2, 0)
    )
    expected_shifts = np.loadtxt(fixture_dir / "local_shifts.txt")

    assert np.allclose(grid, expected_grid)
    assert np.allclose(shifts, expected_shifts)

