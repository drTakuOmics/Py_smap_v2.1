import tempfile
from pathlib import Path

import numpy as np

from smap_tools_python import read_params_file


def test_read_params_file_sample():
    params, fn = read_params_file(Path('sample_search.par'))
    assert fn == 'search_global'
    assert params['nCores'] == 4
    assert np.allclose(params['defocus'], [4407.0, 3189.0, -55.0])
    assert params['Cs'] == 0.000001
    assert params['Cc'] == 0.0027


