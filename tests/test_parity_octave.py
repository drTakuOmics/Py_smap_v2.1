"""Parity tests: Python port vs the original SMAP v2.1 MATLAB code executed in Octave.

Reference file: tests_port/refs/refs.mat, produced by tests_port/octave/make_reference.m
(see tests_port/README.md). Tests are skipped when the file is absent.
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.fft as sfft
import scipy.io

from smap import ctf as sctf
from smap import grid, io, utils
from smap.search_global import SearchSetup, run_job
from smap.structure import Atoms, electrostatic_potential, finalize_sp

REF = Path(__file__).parent / "refs" / "refs.mat"
pytestmark = pytest.mark.skipif(not REF.exists(), reason="Octave reference not generated")


@pytest.fixture(scope="module")
def ref():
    m = scipy.io.loadmat(REF, squeeze_me=False)["ref"][0, 0]
    return {k: m[k] for k in m.dtype.names}


def close(a, b, rtol=1e-5, atol=1e-6):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def test_crop_or_pad(ref):
    close(utils.crop_or_pad(ref["cop_a2"], [7, 9]), ref["cop_crop2"])
    close(utils.crop_or_pad(ref["cop_a2"], [15, 16], 0), ref["cop_pad2"])
    close(utils.crop_or_pad(ref["cop_a3"], [6, 6, 6]), ref["cop_crop3"])
    close(utils.crop_or_pad(ref["cop_a3"], [12, 12, 12], 2.5), ref["cop_pad3"])


def test_resize_for_fft(ref):
    close(utils.resize_for_fft(ref["rff_in"], "pad", 0), ref["rff_pad"])
    close(utils.resize_for_fft(ref["rff_in"], "crop"), ref["rff_crop"])


def test_rrj_nm(ref):
    close(utils.rrj(np.zeros((9, 9))), ref["rrj9"])
    close(utils.rrj(np.zeros((10, 10))), ref["rrj10"])
    close(utils.rrj(np.zeros((9, 9, 9))), ref["rrj9c"])
    close(utils.rrj(np.zeros((10, 10, 10))), ref["rrj10c"])
    close(utils.nm(ref["nm_in"]), ref["nm_out"])


def test_resize_f(ref):
    close(utils.resize_f(ref["rsz_im"], 0.688), ref["rsz_im_out"], atol=1e-10)
    close(utils.resize_f(ref["rsz_vol"], 0.688), ref["rsz_vol_out"], atol=1e-10)


def test_radialmean_psd_cross(ref):
    o1, o2 = utils.radialmean_2d(ref["rm_in"])
    close(o1, ref["rm_o1"].ravel())
    close(o2, ref["rm_o2"])
    f, im = utils.psd_filter(ref["psd_in"], "sqrt")
    close(f, ref["psd_f"])
    close(im, ref["psd_im"], atol=1e-8)
    close(utils.mask_central_cross(ref["mcc_in"]), ref["mcc_out"], atol=1e-10)


def test_bindata(ref):
    ym, yb = utils.bindata_smap(ref["bd_y"].ravel(), ref["bd_x"].ravel(), ref["bd_e"].ravel())
    close(ym, ref["bd_ym"].ravel())
    close(yb, ref["bd_yb"].ravel())


def test_phase_shifts(ref):
    blob = ref["mav_in"].astype(np.float64)
    close(utils.apply_phase_shifts_3d_real(blob, [1.3, -2.7, 0.4]), ref["aps_out"], atol=1e-9)
    close(utils.apply_phase_shifts_3d_real(blob, [2, -3, 1]), ref["aps_out_int"])


def test_ctf(ref):
    p = dict(Cs=2.7e-3, Cc=2.7e-3, V_acc=300000, deltaE=0.7, a_i=50e-6, aPerPix=1.5, F_abs=0)
    df = ref["ctf_df"]
    close(sctf.ctf(df, [64, 64], p), ref["ctf_out"], atol=1e-7)
    close(sctf.ctf(df, [63, 63], p), ref["ctf_out_odd"], atol=1e-7)


def test_search_grid(ref):
    R, ea = grid.calculate_search_grid("D1", 10, 10)
    Rm = np.transpose(ref["grid_R"], (2, 0, 1))
    close(R, Rm, atol=1e-12)
    close(ea, ref["grid_EA"], atol=1e-12)
    Rc, _ = grid.calculate_search_grid("C1", 10, 10)
    assert Rc.shape[0] == 2 * Rm.shape[0]


def test_assign_jobs(ref):
    close(utils.assign_jobs(100, 7, 3), ref["aj"].ravel())
    close(utils.assign_jobs(100, 7, 7), ref["aj_last"].ravel())


def test_params_file(ref):
    par = Path(__file__).resolve().parents[1] / "examples" / "ribosome_in_situ" / "sample_search.par"
    p, _ = io.read_params_file(par)
    close(p["defocus"], ref["par_defocus"].ravel())
    assert p["aPerPix"] == ref["par_aPerPix"].item()
    assert p["aPerPix_search"] == ref["par_aPerPix_search"].item()
    assert p["nCores"] == ref["par_nCores"].item()
    assert p["arbThr"] == ref["par_arbThr"].item()
    assert p["Cs"] == ref["par_Cs"].item()
    assert p["margin_pix"] == ref["par_margin"].item()
    close(p["mask_params"], ref["par_mask"].ravel())


def test_main_loop(ref):
    """smappoi_search_global.m lines 290-560 on synthetic inputs."""
    SPV = ref["ml_SPV"].astype(np.float32)
    imref = ref["ml_imref"].astype(np.float32)
    fPSD = ref["ml_fPSD"].astype(np.float32)
    dfs = ref["ml_dfs"]
    R = np.transpose(ref["ml_R"], (2, 0, 1))
    N0 = N1 = 96
    h0 = N0 // 2 + 1
    V_F = np.fft.fftshift(sfft.fftn(np.fft.ifftshift(SPV.astype(np.complex64)))).astype(np.complex128)
    c = V_F.shape[0] // 2
    V_F[c, c, c] = 0
    imF = sfft.fftn(np.fft.ifftshift(imref)) / np.float32(96.0)
    p = dict(Cs=2.7e-3, Cc=2.7e-3, V_acc=300000, deltaE=0.7, a_i=50e-6, aPerPix=1.5,
             F_abs=0, nCores=1, arbThr=float(ref["ml_arbThr"].item()))
    C = sctf.ctf(dfs, [N0, N1], p)
    C = np.stack([np.fft.ifftshift(C[:, :, i]) for i in range(2)], 2)
    st = SearchSetup(params=p, out_dir=Path("."), scratch_dir=Path("."), R_full=R,
                     dims_imref=(N0, N1), Npix_im=(N0, N1), imref=imref,
                     imref_F_half=imF[:h0].astype(np.complex64), fPSD_half=fPSD[:h0],
                     CTF_half=np.imag(C)[:h0], V_F=V_F, Npix=32, defocus=dfs,
                     high_thr=99.0, n_samples=1.0)
    for bs in (1, 3):
        out = run_job(st, 1, device="cpu", batch_size=bs, write_files=False)
        close(out["sum1"], ref["ml_sum1"], rtol=1e-4, atol=2e-4)
        close(out["sum2"], ref["ml_sum2"], rtol=1e-4, atol=2e-4)
        close(out["SD"], ref["ml_SD"].ravel(), rtol=1e-5)
        arb_m = ref["ml_arb"]
        a = out["arb"]
        # identical hit lists (values may differ in the last float32 bits)
        key_p = {(int(l), int(q)) for _, l, q in a}
        key_m = {(int(l), int(q)) for _, l, q in arb_m}
        assert len(key_p ^ key_m) <= 2, len(key_p ^ key_m)
        common = sorted(key_p & key_m)
        dp = {(int(l), int(q)): v for v, l, q in a}
        dm = {(int(l), int(q)): v for v, l, q in arb_m}
        close([dp[k] for k in common], [dm[k] for k in common], rtol=1e-4)


def test_potential_and_sp(ref, tmp_path):
    xyz = ref["sp_xyz"]
    names = [str(s).strip() for s in np.atleast_1d(ref["sp_atoms"])]
    if len(names) == 1 and len(names[0]) == xyz.shape[1]:
        names = list(names[0])
    atoms = Atoms(xyz, names, np.zeros(xyz.shape[1]), [""] * xyz.shape[1])
    V = electrostatic_potential(atoms, 1.032, b_arb=0)
    close(V.astype(np.float32), ref["sp_V"], rtol=1e-5, atol=1e-6)

    EPV = (ref["sp_V"].astype(np.float64) + 4.871).astype(np.float32).astype(np.float64)
    SP = finalize_sp(EPV, dict(aPerPix=1.032, edge_max=256, V_acc=300000.0), log=lambda m: None)
    assert SP.shape[0] == int(ref["sp_edge"].item())
    close(SP, ref["sp_SP"], rtol=1e-5, atol=1e-6)
