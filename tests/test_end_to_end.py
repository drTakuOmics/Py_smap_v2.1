"""End-to-end functional tests on synthetic data (see tests/synthetic.py).

1. prepare -> run_job (2 jobs) -> combine: the planted particle is the top highVals.txt
   entry with its exact rotation index and pixel.
2. The full pipeline through clustering, refinement and optimisation, once with v2.1
   behaviour and once with every round-1 option on; the output files must parse the way
   the RNF213 review scripts read them.
"""

from pathlib import Path

import mrcfile
import numpy as np
import pytest

from smap.refine import optimize_particles, refine_particles, select_particles, write_final
from smap.search_global import combine, prepare, run_job
from synthetic import make_tile, write_par


@pytest.mark.parametrize("k,r0,c0", [(137, 72, 72), (1500, 50, 95)])
def test_recovers_inserted_particle(tmp_path, k, r0, c0):
    img = make_tile(tmp_path, k, r0, c0, snr=12.0)
    st = prepare(write_par(tmp_path, img))
    for j in (1, 2):
        run_job(st, j, device="cpu", batch_size=8)
    combine(st)
    hv = np.loadtxt(tmp_path / "out" / "highVals.txt", ndmin=2)
    top = hv[np.argmax(hv[:, 1])]
    assert int(top[0]) == k
    assert (int(top[2]), int(top[3])) == (r0 + 1, c0 + 1)  # 1-based, MATLAB convention


def _rnf213_matrix(path, skiprows=0):
    """Reader semantics of the RNF213 review scripts (rnf213_workflow_v3.matrix)."""
    text = Path(path).read_text().splitlines()[skiprows:]
    lines = [x for x in text if x.strip() and not x.lstrip().startswith("#")]
    return np.loadtxt(lines, ndmin=2) if lines else np.empty((0, 0))


@pytest.mark.parametrize("extra", [
    "",                                                     # v2.1 behaviour
    "oversample 2\ndedupe_orientations 1\nfix_optimize_angle 1\nrefine_local_peak 20\n"
    "fix_cluster_thresholds 1\ntemplate_bfactor 20\n",     # all round-1 options
])
def test_full_pipeline_through_refinement(tmp_path, extra):
    r0, c0 = 50, 95
    img = make_tile(tmp_path, 1500, r0, c0, snr=14.0, phase_scale=0.002)
    st = prepare(write_par(tmp_path, img, extra="optThr 5.0\ninc_degrees 2\n" + extra))
    for j in (1, 2):
        run_job(st, j, device="cpu", batch_size=8)
    res = combine(st)
    parts = refine_particles(st, select_particles(st, res), res["mean"], res["sd"], device="cpu")
    opt = optimize_particles(st, parts, device="cpu")
    write_final(st, opt)

    out = tmp_path / "out"
    lf = np.loadtxt(out / "list_final.txt", ndmin=2)
    best = lf[np.argmax(lf[:, 0])]
    assert (int(best[1]), int(best[2])) == (r0 + 1, c0 + 1)
    top = max(opt, key=lambda o: o["peak_opt"])
    assert abs(top["xy_opt"][0] - (r0 + 1)) < 1 and abs(top["xy_opt"][1] - (c0 + 1)) < 1
    assert top["peak_opt"] > 6
    assert abs(top["xy_orig0_est"][0] - c0) < 1 and abs(top["xy_orig0_est"][1] - r0) < 1

    # files must parse the way the RNF213 review scripts read them
    n = len(opt)
    assert _rnf213_matrix(out / "list_final.txt").shape == (n, 10)
    assert _rnf213_matrix(out / "particles.par", skiprows=1).shape == (n, 16)
    assert _rnf213_matrix(out / "particles.txt").shape == (n, 2)
    with mrcfile.open(out / "particles.mrc") as m:
        assert m.data.shape[0] == n
    with mrcfile.open(out / "templates.mrc") as m:
        assert m.data.shape[0] == n
    with mrcfile.open(out / "search_mean.mrc") as m:
        assert tuple(m.data.shape[-2:]) == st.Npix_im
    assert (out / "particles.csv").exists() and (out / "particles_schema.json").exists()
