"""Parity tests for the refinement stage vs the original MATLAB code run in Octave
(tests_port/refs/refs_refine.mat from tests_port/octave/make_reference_refine.m)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.fft as sfft
import scipy.io

from smap import ctf as sctf
from smap.grid import q2r
from smap.refine import _ctf_corner_imag, _Optimizer, _Scorer, cluster_im_by_thr, fminsearch
from smap.rot import bump_rotations, pairwise_qd, rotm_to_quat, smap2frealign
from smap.utils import apply_phase_shifts_2d, crop_patch, ftj, imresize_bilinear, rrj

REF = Path(__file__).parent / "refs" / "refs_refine.mat"
pytestmark = pytest.mark.skipif(not REF.exists(), reason="Octave reference not generated")
PAR = dict(Cs=2.7e-3, Cc=2.7e-3, V_acc=300000, deltaE=0.7, a_i=50e-6, aPerPix=1.5, F_abs=0)


@pytest.fixture(scope="module")
def ref():
    m = scipy.io.loadmat(REF)["ref"][0, 0]
    return {k: m[k] for k in m.dtype.names}


def close(a, b, rtol=1e-5, atol=1e-6):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


def test_cluster(ref):
    R = np.transpose(ref["cl_R"], (2, 0, 1))
    cl = cluster_im_by_thr(ref["cl_im"], ref["cl_mipi"], 2.5, R)
    close([c["xy"] for c in cl], ref["cl_xy"])
    close([c["MaxVal"] for c in cl], ref["cl_mv"].ravel())
    close(np.array([c["q"] for c in cl]), np.transpose(ref["cl_q"], (2, 0, 1)), atol=1e-12)


def test_crop_and_shifts(ref):
    img = ref["cr_img"]
    close(crop_patch(img, (32, 32), (10, 85)), ref["cr_spill"])
    close(crop_patch(img, (32, 32), (50, 40)), ref["cr_in"])
    close(crop_patch(img, (40, 40), (90, 3)), ref["cr_spill2"])
    close(apply_phase_shifts_2d(img, [1.3, -2.6]), ref["aps_real"], atol=1e-9)
    close(apply_phase_shifts_2d(ftj(img), [0.7, -1.2]), ref["aps_cplx"], atol=1e-9)
    close(q2r([1.02, 0.01, -0.03, 0.002]), ref["q2R"], atol=1e-12)


def test_stage_b_scores(ref):
    N = 32
    SPV = ref["b_SPV"].astype(np.float32)
    V_F = np.fft.fftshift(sfft.fftn(np.fft.ifftshift(SPV.astype(np.complex64)))).astype(np.complex128)
    V_F[N // 2, N // 2, N // 2] = 0
    st = SimpleNamespace(V_F=V_F, Npix=N)
    sc = _Scorer(st, ref["b_fPSD"], "cpu")
    xy = tuple(int(v) for v in ref["b_xy"].ravel())
    sc.set_patch(crop_patch(ref["b_imref"], (N, N), xy), crop_patch(ref["b_mean"], (N, N), xy),
                 crop_patch(ref["b_SD"], (N, N), xy))
    R = np.transpose(ref["b_R"], (2, 0, 1))
    dfs = ref["b_dfs"]
    s = sc.score(sc.proj_corner(R), _ctf_corner_imag(dfs[0], N, PAR))
    close(s, ref["b_ccR"].ravel(), rtol=2e-4, atol=2e-4)
    P2 = sc.proj_corner(R[1:2])
    sd = sc.score(P2.expand(3, -1, -1), _ctf_corner_imag(dfs, N, PAR))
    close(sd, ref["b_ccD"].ravel(), rtol=2e-4, atol=2e-4)


def test_stage_c_template_and_objective(ref):
    import torch
    N = 32
    o = _Optimizer.__new__(_Optimizer)
    o.t, o.dev, o.N, o.cp = torch, torch.device("cpu"), N, N // 2 + 1
    SPV = ref["c_SPV"].astype(np.float32)
    V_F = np.fft.fftshift(sfft.fftn(np.fft.ifftshift(SPV)))
    V_F[N // 2, N // 2, N // 2] = 0
    o.Vr = torch.as_tensor(V_F.real.astype(np.float32)).double()
    o.Vi = torch.as_tensor(V_F.imag.astype(np.float32)).double()
    ar = torch.arange(1, N + 1, dtype=torch.float64)
    rr, cc = torch.meshgrid(ar, ar, indexing="ij")
    o.gx, o.gy = cc - o.cp, rr - o.cp
    g = (np.arange(N) - N / 2) / N
    o.gr, o.gc = g[:, None], g[None, :]
    o.inds_mask = (rrj(np.ones((N, N))) * N) > (N / 2)
    CTF = np.fft.ifftshift(sctf.ctf(np.atleast_2d(ref["c_df"].ravel()), [N, N], PAR)[:, :, 0])
    t = o.template(ref["c_Rref"], CTF)
    bg = np.median(t[o.inds_mask])
    close(bg, ref["c_bg"].item(), rtol=1e-4)
    close(t - bg, ref["c_template"], rtol=1e-4, atol=1e-6)
    tF = ftj(t - bg) * ref["c_fP"]
    tF = tF / np.std(tF, ddof=1)
    from smap.utils import iftj
    cc_corr = (iftj(ftj(ref["c_imr"]) * np.conj(tF)) - ref["c_mref"]) / ref["c_sref"]
    cF = ftj(cc_corr)
    vals = [o.cc_value_at_shift(cF, s) for s in ref["c_shs"]]
    # Octave ran this chain in single precision (fPSD_patch is single); we use double.
    close(vals, ref["c_vals"].ravel(), rtol=5e-4, atol=1e-4)


def test_rotation_helpers():
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    Q *= np.sign(np.linalg.det(Q))
    q = rotm_to_quat(Q)
    w, x, y, z = q
    Rq = np.array([[w*w + x*x - y*y - z*z, 2*(x*y - w*z), 2*(x*z + w*y)],
                   [2*(x*y + w*z), w*w - x*x + y*y - z*z, 2*(y*z - w*x)],
                   [2*(x*z - w*y), 2*(y*z + w*x), w*w - x*x - y*y + z*z]])
    close(Rq, Q, atol=1e-12)                      # quaternion.RotationMatrix inverse
    Rz = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])   # 90 deg about z
    close(pairwise_qd(np.eye(3)[None], Rz[None]), [[90.0]], atol=1e-9)
    B = bump_rotations(2.0, 1.0)
    ang = pairwise_qd(np.eye(3)[None], B)[0]
    assert ang.max() <= 2.0 + 1e-9 and np.isclose(ang.min(), 0)
    assert smap2frealign(np.eye(3)[None]).shape == (1, 3)


def test_imresize_matches_matlab_values():
    # MATLAB: imresize(1:8,[1 4],'bilinear') = [1.625 3.5 5.5 7.375]; upsampling 1:4 -> 1x8
    close(imresize_bilinear(np.arange(1, 9, dtype=float)[None], (1, 4)), [[1.625, 3.5, 5.5, 7.375]])
    close(imresize_bilinear(np.array([[1.0, 2, 3, 4]]), (1, 8)),
          [[1, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4]])


def test_fminsearch_rosenbrock():
    f = lambda v: 100 * (v[1] - v[0] ** 2) ** 2 + (1 - v[0]) ** 2  # noqa: E731
    x, fv = fminsearch(f, [-1.2, 1.0], tol_fun=1e-4, tol_x=1e-4)
    close(x, [1.0, 1.0], atol=1e-3)            # MATLAB doc example converges to [1 1]
