import numpy as np
import numpy as np
from smap_tools_python import cluster_im_by_thr


def test_cluster_im_by_thr_basic():
    im = np.zeros((5, 5), dtype=float)
    idx = np.zeros((5, 5), dtype=int)
    rotations = ['r0', 'r1']
    # cluster 1
    im[1, 1] = 1
    im[1, 2] = 1
    idx[1, 1] = 0
    idx[1, 2] = 0
    # cluster 2 with higher score
    im[3, 3] = 2
    im[3, 4] = 2
    im[4, 3] = 2
    idx[3, 3] = 1
    idx[3, 4] = 1
    idx[4, 3] = 1
    ss, q_best, order, xy = cluster_im_by_thr(im, idx, 0.5, rotations)
    assert len(ss) == 2
    assert np.array_equal(order, [1, 0])
    assert q_best[0] == 'r1'
    assert tuple(xy[0]) == ss[0]['xy']
    assert ss[0]['MaxVal'] == 2
