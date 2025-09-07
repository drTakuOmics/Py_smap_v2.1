from pathlib import Path

import numpy as np

from smap_tools_python import smappoi_search_global, write_mrc


def test_smappoi_search_global_stdout(tmp_path, capsys):
    arr = np.zeros((8, 8), dtype=np.float32)
    arr[2:4, 2:4] = 1.0
    img = tmp_path / 'image.mrc'
    model = tmp_path / 'model.mrc'
    write_mrc(img, arr)
    write_mrc(model, arr)

    par_template = Path('tests/fixtures/simple_search.par').read_text()
    par = tmp_path / 'search.par'
    par.write_text(par_template.format(imageFile=img, modelFile=model))

    smappoi_search_global(par)
    captured = capsys.readouterr()
    expected = Path('tests/fixtures/simple_stdout.txt').read_text().strip()
    assert captured.out.strip() == expected
