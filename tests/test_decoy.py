"""Target-decoy calibration: decoy inputs and the FDR / q-value bookkeeping."""

import json

import numpy as np

from smap import io as sio
from smap.decoy import calibrate, count_ge, fdr_curve, main as decoy_main, q_values
from smap.search_global import apply_decoy, main
from synthetic import make_tile


def _log(*a, **k):
    pass


def test_mirror_decoy_keeps_centre_voxel(tmp_path):
    for n in (7, 8):
        v = np.zeros((n, n, n), np.float32)
        cp = n // 2
        v[cp, cp, cp] = 5.0
        v[cp, cp, cp + 2] = 1.0                       # off-centre marker along the page axis
        sio.mw(v, tmp_path / f"v{n}.mrc", 1.0)
        p = dict(modelFile=str(tmp_path / f"v{n}.mrc"), aPerPix=1.0, decoy_template="mirror")
        apply_decoy(p, tmp_path, _log)
        m = sio.mr(p["modelFile"])
        assert m[cp, cp, cp] == 5.0                   # rotation centre unchanged
        assert m[cp, cp, cp - 2] == 1.0               # page axis reflected about the centre
        assert p["decoy"]["template"] == "mirror"


def test_phase_random_image_keeps_amplitudes(tmp_path):
    rng = np.random.default_rng(3)
    im = rng.standard_normal((64, 80)).astype(np.float32) + 2.0
    sio.mw(im, tmp_path / "im.mrc", 1.0)
    p = dict(imageFile=str(tmp_path / "im.mrc"), aPerPix=1.0, decoy_image="phase_random", decoy_seed=4)
    apply_decoy(p, tmp_path, _log)
    out = sio.mr(p["imageFile"]).astype(np.float64)
    a, b = np.abs(np.fft.fft2(im.astype(np.float64))), np.abs(np.fft.fft2(out))
    assert np.allclose(a, b, rtol=1e-4, atol=1e-3 * a.max())
    assert abs(np.corrcoef(im.ravel(), out.ravel())[0, 1]) < 0.1


def test_fdr_and_q_values():
    t = np.array([9.0, 8.5, 8.0, 7.5, 7.0, 6.5])
    d = np.array([7.2, 6.6])
    cut = np.array([6.5, 7.0, 7.5, 8.0])
    nt, nd, fdr, q = fdr_curve(t, d, cut)
    assert nt.tolist() == [6, 5, 4, 3] and nd.tolist() == [2, 1, 0, 0]
    assert np.allclose(fdr, [2 / 6, 1 / 5, 0, 0]) and np.all(np.diff(q) <= 0)
    assert count_ge(t, np.array([8.0]))[0] == 3        # >= as in the RNF213 review
    qv = q_values(t, d)
    assert qv[0] == 0 and qv[-1] == np.max(qv)


def test_target_vs_mirror_decoy_pipeline(tmp_path):
    make_tile(tmp_path, 1500, 50, 95, snr=14.0, phase_scale=0.002)
    common = ("function search_global\nnCores 1\n"
              f"imageFile {tmp_path}/img.mrc\nstructureFile {tmp_path}/vol.mrc\n"
              "aPerPix 2.0\ndefocus 15000.0 14000.0 30.0\nangle_inc 20\nT_sample 0\n"
              "arbThr 4.0\nmargin_pix 0\noptThr 5.0\ninc_degrees 2\n")
    (tmp_path / "t.par").write_text(common + f"outputDir {tmp_path}/target\n")
    (tmp_path / "d.par").write_text(common + f"outputDir {tmp_path}/decoy\ndecoy_template mirror\n")
    assert main([str(tmp_path / "t.par"), "--device", "cpu", "--batch-size", "8"]) == 0
    assert main([str(tmp_path / "d.par"), "--device", "cpu", "--batch-size", "8"]) == 0
    st = json.loads((tmp_path / "decoy" / "search_status.json").read_text())
    assert st["decoy"]["template"] == "mirror"

    # planted particle: final score ~7.2 at this SNR; mirror decoy tops out near 5.5
    rep = calibrate([(tmp_path / "target", tmp_path / "decoy")], "final", report_cutoff=6.5)
    tmax = max(np.loadtxt(tmp_path / "target" / "particles.txt", ndmin=2)[:, 1])
    dmax = max(np.loadtxt(tmp_path / "decoy" / "particles.txt", ndmin=2)[:, 1])
    assert tmax > dmax + 1.0
    assert rep["at_cutoff"]["n_target"] > rep["at_cutoff"]["n_decoy"]
    assert decoy_main(["--pair", str(tmp_path / "target"), str(tmp_path / "decoy"),
                       "--cutoff", "6.5", "--out", str(tmp_path / "cal")]) == 0
    assert (tmp_path / "cal.json").exists() and (tmp_path / "cal.csv").exists()
