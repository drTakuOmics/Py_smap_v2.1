"""Full-image MIP maps and the p-value metric (new outputs; v2.1 files unchanged)."""

import json

import numpy as np

from smap import io as sio
from smap.pvalue import neg_log_p, probit
from smap.search_global import combine, prepare, run_job
from synthetic import make_tile, write_par


def test_whitened_sector_formula():
    """p = exp(-r^2/2) gamma/2pi equals the Gaussian mass beyond r inside the quadrant."""
    rng = np.random.default_rng(4)
    C = np.array([[1.0, 0.6], [0.6, 1.0]])
    L = np.linalg.cholesky(C)
    s = rng.standard_normal((400000, 2)) @ L.T
    Ci = np.linalg.inv(C)
    for x in ([1.5, 2.0], [2.5, 1.0]):
        x = np.array(x)
        r2 = x @ Ci @ x
        mc = np.mean((s[:, 0] > 0) & (s[:, 1] > 0) & (np.einsum("ni,ij,nj->n", s, Ci, s) >= r2))
        W = np.linalg.inv(np.linalg.cholesky(C))           # any whitening gives the same angle
        a, b = W[:, 0], W[:, 1]
        g = np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
        assert abs(np.exp(-r2 / 2) * g / (2 * np.pi) - mc) < 3e-3


def test_probit_and_noise_tail():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(5000)
    p = probit(x)
    assert np.allclose(np.sort(p), np.sort(probit(np.arange(5000))))     # rank-only
    z = rng.standard_normal(40000)
    m = 0.7 * z + 0.7 * rng.standard_normal(40000)
    nl = neg_log_p(m, z)
    # pure background: P(-ln p > c) <= ~exp(-c); 40000 pixels -> very few above 12
    assert np.sum(nl > 12) <= 2


def test_mip_maps_match_v21_values_and_rank_planted_particle(tmp_path):
    r0, c0, k = 50, 95, 1500
    img = make_tile(tmp_path, k, r0, c0, snr=12.0)
    st = prepare(write_par(tmp_path, img))
    for j in (1, 2):
        run_job(st, j, device="cpu", batch_size=8)
    combine(st)
    out = tmp_path / "out"
    mip_s = sio.mr(out / "search_mip_scaled.mrc")
    vals = sio.mr(out / "search_vals.mrc")
    vals = vals if vals.ndim == 2 else vals.max(axis=2)
    kept = vals > 0                                   # v2.1 keeps flat-fielded values above arbThr
    assert kept.any()
    assert np.allclose(mip_s[kept], vals[kept], atol=2e-4)
    q = sio.mr(out / "search_mip_qInds.mrc")
    assert int(q[r0, c0]) == k
    pv = sio.mr(out / "search_pval.mrc")
    assert np.unravel_index(np.argmax(pv), pv.shape) == (r0, c0)
    st_json = json.loads((out / "search_status.json").read_text())
    assert st_json["mip"]["max_neg_log_p"] > 8
