"""End-to-end functional test: prepare -> run_job (2 jobs) -> combine on synthetic data.

A random atom-like volume is projected at a known grid orientation, CTF-filtered and
inserted at a known pixel position in white noise. The search must return that
rotation index and position as the top entry of highVals.txt.
"""

import numpy as np
import pytest
from pathlib import Path

from smap import io as sio
from smap.ctf import ctf
from smap.search_global import combine, make_templates, prepare, run_job

N = 48
IM = 144


def _volume(rng):
    z, y, x = np.mgrid[:N, :N, :N].astype(np.float64)
    pts = rng.uniform(-1, 1, (4000, 3))
    pts = pts[np.linalg.norm(pts, axis=1) < 1][:400] * 15 + 24
    vol = np.zeros((N, N, N), np.float32)
    for pz, py, px in pts:
        vol += np.exp(-((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) / 1.5).astype(np.float32)
    return vol


def _write_par(d, img):
    p = d / "s.par"
    p.write_text(
        "function search_global\n"
        "nCores 2\n"
        f"imageFile {img}\n"
        f"structureFile {d / 'vol.mrc'}\n"
        f"outputDir {d / 'out'}\n"
        "aPerPix 2.0\n"
        "defocus 15000.0 14000.0 30.0\n"
        "angle_inc 20\n"
        "T_sample 0\n"
        "arbThr 4.0\n"
        "margin_pix 0\n"
    )
    return p


@pytest.mark.parametrize("k,r0,c0", [(137, 72, 72), (1500, 50, 95)])
def test_recovers_inserted_particle(tmp_path, k, r0, c0):
    rng = np.random.default_rng(0)
    sio.mw(np.transpose(_volume(rng), (1, 2, 0)), tmp_path / "vol.mrc", 2.0)
    sio.mw(rng.standard_normal((IM, IM)).astype(np.float32), tmp_path / "dummy.mrc", 2.0)
    st0 = prepare(_write_par(tmp_path, tmp_path / "dummy.mrc"), write_files=False)

    tpl = make_templates(st0, [k])[0]
    img = np.zeros((IM, IM))
    h = N // 2
    img[r0 - h:r0 - h + N, c0 - h:c0 - h + N] = tpl
    Ci = np.imag(np.fft.ifftshift(ctf(st0.defocus, [IM, IM], st0.params)[:, :, 0]))
    sig = np.real(np.fft.ifft2(np.fft.fft2(img) * Ci))
    sig *= 12.0 / np.linalg.norm(sig)
    sio.mw((sig + rng.standard_normal((IM, IM))).astype(np.float32), tmp_path / "img.mrc", 2.0)

    st = prepare(_write_par(tmp_path, tmp_path / "img.mrc"))
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
    """prepare -> search -> combine -> cluster -> refine -> optimise -> final files."""
    from smap.refine import optimize_particles, refine_particles, select_particles, write_final

    k, r0, c0 = 1500, 50, 95
    rng = np.random.default_rng(0)
    # potential scaled to small phase shifts so the exit-wave template is in the weak-phase regime
    sio.mw(np.transpose(_volume(rng) * 0.002, (1, 2, 0)), tmp_path / "vol.mrc", 2.0)
    sio.mw(rng.standard_normal((IM, IM)).astype(np.float32), tmp_path / "dummy.mrc", 2.0)
    st0 = prepare(_write_par(tmp_path, tmp_path / "dummy.mrc"), write_files=False)
    tpl = make_templates(st0, [k])[0]
    img = np.zeros((IM, IM))
    h = N // 2
    img[r0 - h:r0 - h + N, c0 - h:c0 - h + N] = tpl
    Ci = np.imag(np.fft.ifftshift(ctf(st0.defocus, [IM, IM], st0.params)[:, :, 0]))
    sig = np.real(np.fft.ifft2(np.fft.fft2(img) * Ci))
    sig *= 14.0 / np.linalg.norm(sig)
    sio.mw((sig + rng.standard_normal((IM, IM))).astype(np.float32), tmp_path / "img.mrc", 2.0)
    par = _write_par(tmp_path, tmp_path / "img.mrc")
    par.write_text(par.read_text() + "optThr 5.0\ninc_degrees 2\n" + extra)
    st = prepare(par)
    for j in (1, 2):
        run_job(st, j, device="cpu", batch_size=8)
    res = combine(st)
    jobs = select_particles(st, res)
    parts = refine_particles(st, jobs, res["mean"], res["sd"], device="cpu")
    opt = optimize_particles(st, parts, device="cpu")
    write_final(st, opt)
    lf = np.loadtxt(tmp_path / "out" / "list_final.txt", ndmin=2)
    best = lf[np.argmax(lf[:, 0])]
    assert (int(best[1]), int(best[2])) == (r0 + 1, c0 + 1)
    top = max(opt, key=lambda o: o["peak_opt"])
    assert abs(top["xy_opt"][0] - (r0 + 1)) < 1 and abs(top["xy_opt"][1] - (c0 + 1)) < 1
    assert top["peak_opt"] > 6
    for f in ("particles.txt", "particles.par", "particles.csv", "particles.mrc", "templates.mrc"):
        assert (tmp_path / "out" / f).exists()
    # files must parse the way the RNF213 review scripts read them
    out = tmp_path / "out"
    n = len(opt)
    assert _rnf213_matrix(out / "list_final.txt").shape == (n, 10)
    assert _rnf213_matrix(out / "particles.par", skiprows=1).shape == (n, 16)
    assert _rnf213_matrix(out / "particles.txt").shape == (n, 2)
    import mrcfile
    with mrcfile.open(out / "particles.mrc") as m:
        assert m.data.shape[0] == n
    with mrcfile.open(out / "search_mean.mrc") as m:
        assert tuple(m.data.shape[-2:]) == st.Npix_im
