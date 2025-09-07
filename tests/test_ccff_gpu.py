import numpy as np
from smap_tools_python import ccff_gpu, ccff


def test_ccff_gpu_matches_cpu():
    rng = np.random.default_rng(0)
    im = rng.standard_normal((16, 16))
    temps = rng.standard_normal((16, 16, 2))
    out_gpu, peaks_gpu = ccff_gpu(im, temps)
    out_cpu, peaks_cpu = ccff(im, temps)
    assert np.allclose(out_gpu, out_cpu)
    assert np.allclose(peaks_gpu, peaks_cpu)
