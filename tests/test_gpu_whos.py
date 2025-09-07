import numpy as np
from smap_tools_python import gpu_whos


def test_gpu_whos_empty():
    local_array = np.ones(5)
    summary, info, total = gpu_whos()
    assert summary == ""
    assert info == []
    assert total == 0
