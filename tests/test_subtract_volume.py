import numpy as np
from smap_tools_python import subtract_volume


def test_subtract_volume_reduces_correlation():
    rng = np.random.default_rng(0)
    N = 32
    x = np.linspace(-1, 1, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    template = np.exp(-(X**2 + Y**2 + Z**2) / 0.1)
    rec = template + 0.1 * rng.standard_normal(template.shape)
    out, scaled = subtract_volume(rec, template)
    corr_before = np.corrcoef(rec.ravel(), template.ravel())[0, 1]
    corr_after = np.corrcoef(out.ravel(), template.ravel())[0, 1]
    assert corr_after < corr_before
    out2, _ = subtract_volume(rec, template, "twofold")
    assert np.allclose(out2, out - scaled)
