import numpy as np
import pandas as pd
from pathlib import Path

from smap_tools_python import smappoi_search_local
from smap_tools_python.rotations_io import read_rotations_file


def test_smappoi_search_local(tmp_path):
    fixture_dir = Path('tests/fixtures')
    params_file = tmp_path / "params.par"
    params_file.write_text(
        f"""
function search_local
rotationsFile {fixture_dir / 'local_rotations.txt'}
tableFile {fixture_dir / 'local_table.csv'}
"""
    )

    df, rotations = smappoi_search_local(params_file)
    expected_df = pd.read_csv(fixture_dir / 'local_table.csv')
    expected_rot = read_rotations_file(fixture_dir / 'local_rotations.txt')

    pd.testing.assert_frame_equal(df, expected_df)
    assert np.allclose(rotations, expected_rot)
