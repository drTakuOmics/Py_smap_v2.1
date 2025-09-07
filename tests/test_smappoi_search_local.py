import numpy as np
from pathlib import Path

from smap_tools_python import smappoi_search_local


def test_smappoi_search_local_grid_and_shifts(tmp_path, capsys):
    fixture_dir = Path("tests/fixtures")

    par_template = (fixture_dir / "local_search.par").read_text()
    rot_file = (fixture_dir / "local_rotations.txt").resolve()
    params_file = tmp_path / "params.par"
    params_file.write_text(par_template.format(rotationsFile=rot_file))

    grid, shifts = smappoi_search_local(params_file)
    captured = capsys.readouterr()
    expected_stdout = (fixture_dir / "local_stdout.txt").read_text().strip()
    assert captured.out.strip() == expected_stdout

    expected_grid = (
        np.loadtxt(fixture_dir / "local_grid.txt")
        .reshape(-1, 3, 3)
        .transpose(1, 2, 0)
    )
    expected_shifts = np.loadtxt(fixture_dir / "local_shifts.txt")

    assert np.allclose(grid, expected_grid)
    assert np.allclose(shifts, expected_shifts)

