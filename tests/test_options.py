"""Tests for the opt-in improvements (defaults reproduce v2.1; see test_parity_*)."""

import json
import math

import mrcfile
import numpy as np
import scipy.fft as sfft

from smap import io as sio
from smap.ctf import template_envelope
from smap.grid import calculate_search_grid
from smap.rot import pairwise_qd
from smap.search_global import SearchSetup, combine, make_templates, prepare, run_job
from smap.stats import cluster_survival, exact_survival, fit_neff
from smap.structure import Atoms, electrostatic_potential
from smap.utils import crop_or_pad, matlab_mode, normalize_rm


def _atoms_volume(N, n_atoms=600, s=1.2, seed=1):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, (20 * n_atoms, 3))
    pts = pts[np.linalg.norm(pts, axis=1) < 1][:n_atoms] * (N * 0.3)
    cp = N // 2 + 1
    ax = np.arange(1, N + 1) - cp
    Y, X, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    vol = np.zeros((N, N, N))
    for x, y, z in pts:
        vol += np.exp(-((X - x) ** 2 + (Y - y) ** 2 + (Z - z) ** 2) / (2 * s * s))
    return vol, pts, ax


def _setup_from_volume(vol, K, R):
    N = vol.shape[0]
    v = vol.astype(np.float32)
    if K > 1:
        v = crop_or_pad(v - matlab_mode(v), [K * N] * 3, 0).astype(np.float32)
    V_F = np.fft.fftshift(sfft.fftn(np.fft.ifftshift(v.astype(np.complex64)))).astype(np.complex128)
    c = V_F.shape[0] // 2
    V_F[c, c, c] = 0
    return SearchSetup(params={}, out_dir=None, scratch_dir=None, R_full=R, dims_imref=(N, N),
                       Npix_im=(N, N), imref=None, imref_F_half=None, fPSD_half=None,
                       CTF_half=None, V_F=V_F, Npix=N, defocus=np.zeros((1, 3)), high_thr=0,
                       n_samples=0, oversample=K)


def _corr(a, b):
    A, B = np.fft.fft2(a), np.fft.fft2(b)
    A[0, 0] = B[0, 0] = 0
    return float(np.real(np.sum(A * np.conj(B))) / np.sqrt(np.sum(np.abs(A) ** 2) * np.sum(np.abs(B) ** 2)))


def test_oversampling_improves_template_accuracy():
    N, s = 64, 1.2
    vol, pts, ax = _atoms_volume(N, s=s)
    rng = np.random.default_rng(4)
    R = normalize_rm(np.array([np.linalg.qr(rng.standard_normal((3, 3)))[0] for _ in range(4)]))
    R = np.array([r * np.sign(np.linalg.det(r)) for r in R])
    exact = []
    for r in R:                       # analytic projection of Gaussian atoms
        u, v = pts @ r[0], pts @ r[1]
        exact.append(sum(np.exp(-((ax[None, :] - uu) ** 2 + (ax[:, None] - vv) ** 2) / (2 * s * s))
                         for uu, vv in zip(u, v)))
    cc = {}
    for K in (1, 2):
        T = make_templates(_setup_from_volume(vol, K, R), np.arange(1, 5))
        cc[K] = np.mean([_corr(T[i], exact[i]) for i in range(4)])
    assert cc[2] > cc[1]
    assert cc[2] > 0.999


def test_dedupe_orientations_removes_only_duplicates():
    R0, _ = calculate_search_grid("C1", 20, 20)
    R1, _ = calculate_search_grid("C1", 20, 20, dedupe=True)
    assert R1.shape[0] < R0.shape[0]
    d = pairwise_qd(R1, R1)
    np.fill_diagonal(d, np.inf)
    assert d.min() > 1e-3                     # no identical orientations left (degrees)
    d01 = pairwise_qd(R0, R1).min(axis=1)
    assert d01.max() < 1e-4                   # every original orientation still represented


def test_fix_even_grid_spacing():
    a = 1.0
    xyz = np.array([[104.4, -104.4], [0.0, 0.0], [0.0, 0.0]])   # -> even box (214)
    at = Atoms(xyz, ["C", "C"], np.zeros(2), ["A", "A"])
    for exact, expect in ((False, 104.4 * 214 / 213), (True, 104.4)):
        V = electrostatic_potential(at, a, exact_spacing=exact)
        assert V.shape[0] == 214
        prof = V.sum(axis=(0, 2))             # along columns = x
        cp = 214 // 2 + 1
        idx = np.arange(1, 215) - cp
        right = idx > 0
        com = np.sum(idx[right] * prof[right]) / np.sum(prof[right])
        assert abs(com - expect) < 0.05, (exact, com, expect)


def test_template_envelope():
    p = dict(aPerPix=1.5)
    assert np.all(template_envelope(64, p) == 1)
    e = template_envelope(64, dict(aPerPix=1.5, template_bfactor=100))
    k_nyq = 1 / (2 * 1.5)
    assert np.isclose(e[32, 32], 1) and np.isclose(e[32, 0], math.exp(-100 * k_nyq**2 / 4))


def test_exact_survival_is_strict():
    s = [0.0, 1.0, 2.0, 2.0]
    assert list(exact_survival(s, [-1, 0, 1, 1.9998, 2, 3])) == [4, 3, 2, 2, 0, 0]


def test_cluster_neff_detects_correlation():
    rng = np.random.default_rng(0)
    from scipy import ndimage
    white = rng.standard_normal((512, 512, 4))
    smooth = np.stack([ndimage.gaussian_filter(rng.standard_normal((512, 512)), 3) for _ in range(4)], 2)
    smooth /= smooth.std()
    xs = np.linspace(2.0, 3.5, 12)
    n = white.size
    fw = fit_neff(xs, cluster_survival(white, xs))
    fs = fit_neff(xs, cluster_survival(smooth, xs))
    assert 0.3 < fw["N_eff"] / n < 1.2          # mostly single-pixel clusters
    assert fs["N_eff"] / n < 0.1 * fw["N_eff"] / n   # correlated field: far fewer clusters


def test_layout_adapters_roundtrip(tmp_path):
    v = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)   # asymmetric [row,col,page]
    sio.mw(v, tmp_path / "v.mrc", 1.0)
    with mrcfile.open(tmp_path / "v.mrc") as m:
        native = np.array(m.data)                                  # [z, y, x]
    assert native.shape == (4, 2, 3)
    assert np.array_equal(sio.smap_to_zyx(v), native)
    assert np.array_equal(sio.zyx_to_smap(native), v)
    assert np.array_equal(sio.mr(tmp_path / "v.mrc"), v)


def test_incomplete_run_is_flagged(tmp_path):
    from synthetic import IM, volume, write_par
    rng = np.random.default_rng(0)
    sio.mw(np.transpose(volume(rng), (1, 2, 0)), tmp_path / "vol.mrc", 2.0)
    sio.mw(rng.standard_normal((IM, IM)).astype(np.float32), tmp_path / "img.mrc", 2.0)
    st = prepare(write_par(tmp_path, tmp_path / "img.mrc"))
    for j in (1, 2):
        run_job(st, j, device="cpu", batch_size=8, max_rotations=50)
    combine(st)
    status = json.loads((tmp_path / "out" / "search_status.json").read_text())
    assert status["is_complete"] is False and status["evaluated_pose_count"] == 100
