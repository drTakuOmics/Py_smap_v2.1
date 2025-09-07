import numpy as np
import pandas as pd

from smap_tools_python import write_rotations_file, smappoi_search_local


def test_smappoi_search_local(tmp_path):
    rot = np.stack([np.eye(3), np.eye(3)], axis=2)
    rot_file = tmp_path / "rots.txt"
    write_rotations_file(rot, rot_file)

    table = pd.DataFrame({"fn_patches": ["p1.mrc", "p2.mrc"], "patches_pageref": [1, 2]})
    table_file = tmp_path / "table.csv"
    table.to_csv(table_file, index=False)

    params_file = tmp_path / "params.par"
    params_file.write_text(
        f"""
function search_local
rotationsFile {rot_file}
tableFile {table_file}
"""
    )

    df, rotations = smappoi_search_local(params_file)
    assert list(df.columns) == ["fn_patches", "patches_pageref"]
    assert len(df) == 2
    assert rotations.shape == (3, 3, 2)
