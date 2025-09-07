import numpy as np
from smap_tools_python import pr_quick


def test_pr_quick_basic():
    rng = np.random.default_rng(0)
    vals = rng.normal(2, 1, 1000)
    vals_ctrl = rng.normal(0, 1, 1000)
    table = {"peak_part": vals, "peak_part_ctrl": vals_ctrl}
    thr_ref, prec, recall, thr_F1, F1_opt = pr_quick(table)
    assert len(thr_ref) <= 4
    assert np.all(np.diff(recall) <= 1e-9)
    assert 0 <= F1_opt <= 1
    assert thr_F1 >= vals_ctrl.min() and thr_F1 <= vals.max()
