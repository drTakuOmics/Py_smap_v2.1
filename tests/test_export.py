"""RELION export: Euler convention checked against an independent real-space projector."""

import numpy as np
import pytest
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from smap import io as sio
from smap.export import (frealign_to_smap, main as export_main, read_star_particles, relion_angles,
                         relion_matrix)
from smap.rot import smap2frealign
from smap.search_global import make_templates, prepare
from synthetic import N, make_tile, write_par


def _project_relion(vol_smap, A):
    """RELION definition, real space: P(x, y) = sum_z V(A^T [x, y, z]); x = column, y = row,
    origin at N//2. Independent of SMAP's Fourier-slice code."""
    n = vol_smap.shape[0]
    c = n // 2
    g = np.arange(n) - c
    y, x, z = np.meshgrid(g, g, g, indexing="ij")              # output row, col, depth
    xyz = np.stack([x, y, z]).reshape(3, -1)
    src = A.T @ xyz + c                                          # (x', y', z') in volume
    v = map_coordinates(vol_smap, [src[1], src[0], src[2]], order=1, mode="constant")
    return v.reshape(n, n, n).sum(axis=2)


def _ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def test_relion_angles_round_trip():
    Rs = Rotation.random(200, random_state=2).as_matrix()
    Rs = np.concatenate([Rs, [np.eye(3), np.diag([-1.0, 1, -1]), np.diag([1.0, -1, -1])]])
    for R in Rs:
        assert np.allclose(relion_matrix(*relion_angles(R)), R, atol=1e-9)


def test_frealign_inverse():
    Rs = Rotation.random(50, random_state=3).as_matrix()
    D = np.diag([1.0, -1, -1])
    for R in Rs:
        psi, th, phi = smap2frealign((R @ D)[None])[0]          # as written to particles.par
        assert np.allclose(frealign_to_smap(psi, th, phi), R, atol=1e-6)


def test_relion_projection_matches_smap_template(tmp_path):
    make_tile(tmp_path, 1, 72, 72, snr=1.0)
    st = prepare(write_par(tmp_path, tmp_path / "img.mrc"), write_files=False)
    vol = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(st.V_F))))
    assert vol.shape == (N, N, N)
    ang = np.degrees(np.arccos(np.clip((np.trace(st.R_full, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    pool = np.nonzero((ang > 40) & (ang < 140))[0]       # R != R^T, so the inverse is a distinct pose
    rng = np.random.default_rng(5)
    for k in rng.choice(pool, 4, replace=False) + 1:
        R = st.R_full[k - 1]
        A = relion_matrix(*relion_angles(R))
        t = make_templates(st, [k])[0]
        good = _ncc(t, _project_relion(vol, A))
        wrong_inv = _ncc(t, _project_relion(vol, A.T))
        wrong_hand = _ncc(t, _project_relion(vol[:, :, ::-1], A))
        print(f"k={k} ncc={good:.4f} inverse={wrong_inv:.4f} mirror={wrong_hand:.4f}")
        assert good > 0.97, good
        assert good - max(wrong_inv, wrong_hand) > 0.1, (good, wrong_inv, wrong_hand)


def test_export_cli_smap(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    R = Rotation.from_euler("zyz", [10, 50, -30], degrees=True).as_matrix()
    hdr = ("num_id,score_final,score_refined,x_orig0_est,y_orig0_est,row,col,row_opt,col_opt,row_search,"
           "col_search,df1_nm,df2_nm,ast_rad,"
           + ",".join(f"q_opt_{i}{j}" for i in range(3) for j in range(3)))
    vals = [1, 9.1, 8.7, 100.25, 40.5, 1, 1, 1, 1, 1, 1, 1500, 1400, -0.5, *R.ravel()]
    (out / "particles.csv").write_text(hdr + "\n" + ",".join(map(str, vals)) + "\n")
    (tmp_path / "search.par").write_text("function search_global\naPerPix 1.5\nV_acc 300000\nCs 0.0027\n")
    sio.mw(np.zeros((32, 32, 1), np.float32), out / "particles.mrc", 1.5)
    assert export_main(["smap", str(out), "--micrograph", "mic.mrc", "--offset", "1000", "2000",
                        "--images", "-o", str(tmp_path / "p.star")]) == 0
    (r,) = read_star_particles(tmp_path / "p.star")
    assert float(r["rlnCoordinateX"]) == pytest.approx(1100.25)
    assert float(r["rlnCoordinateY"]) == pytest.approx(2040.5)
    assert float(r["rlnDefocusU"]) == pytest.approx(15000)
    assert float(r["rlnDefocusAngle"]) == pytest.approx(0.5 * 180 / np.pi, abs=1e-3)
    A = relion_matrix(float(r["rlnAngleRot"]), float(r["rlnAngleTilt"]), float(r["rlnAnglePsi"]))
    assert np.allclose(A, R, atol=1e-5)
    assert r["rlnImageName"].startswith("000001@")
    text = (tmp_path / "p.star").read_text()
    assert "_rlnImageSize #8" in text and " 32 2" in text
