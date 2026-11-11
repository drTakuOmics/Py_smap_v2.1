"""tools/check_reconstruction.py: reconstruction from a STAR file with RELION's conventions (new; not part of v2.1)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import mrcfile
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from smap.export import write_star

ROOT = Path(__file__).resolve().parents[1]
N, PIX = 32, 3.0


def _tool():
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("check_reconstruction", ROOT / "tools" / "check_reconstruction.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _volume(rng, ops=None):
    g = np.arange(N) - N // 2
    z, y, x = np.meshgrid(g, g, g, indexing="ij")                      # mrc layout (z, y, x)
    pts = rng.uniform(-9, 9, (40 if ops is None else 4, 3))
    if ops is not None:                                                # symmetric: V(M x) = V(x)
        pts = np.concatenate([pts @ M.T for M in ops])
    V = np.zeros((N, N, N))
    for px, py, pz in pts:
        V += np.exp(-((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) / 2.0)
    return V, (x, y, z)


def _project(V, grid, A):
    """RELION, real space: P(x, y) = sum_z V(A^T [x, y, z]) (x = column, y = row)."""
    x, y, z = grid
    src = A.T @ np.stack([x, y, z]).reshape(3, -1) + N // 2
    out = map_coordinates(V, [src[2], src[1], src[0]], order=1, mode="constant")   # V[z, y, x]
    return out.reshape(N, N, N).sum(axis=0)


def _dataset(d: Path, mod, n=300, offset=30.0, transpose=False, ops=None):
    rng = np.random.default_rng(3)
    V, grid = _volume(rng, ops)
    rots = Rotation.random(20 * n, random_state=7).as_matrix()
    if ops is not None:                        # as a search with symmetryFile: one asymmetric unit only
        from smap.symmetry import asu_mask
        rots = rots[asu_mask(rots, ops)]
    with mrcfile.new(d / "template.mrc", overwrite=True) as m:
        m.set_data(V.astype(np.float32))
        m.voxel_size = PIX
    f = np.fft.fftfreq(N, d=PIX)
    KY, KX = np.meshgrid(f, f, indexing="ij")
    imgs, rows = [], []
    for i in range(n):
        A = rots[i]
        dfu, dfv, ang = 12000 + 40 * (i % 50), 11500 + 40 * (i % 50), 30.0
        c = mod.ctf_2d(KX, KY, dfu, dfv, ang, 300.0, 2.7, 0.07)
        p = np.real(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(_project(V, grid, A))) * c))
        p = np.fft.fftshift(p)
        p = p / p.std() + 0.7 * rng.standard_normal((N, N)) + offset          # raw counts: large mean
        imgs.append(p.astype(np.float32))
        rows.append(dict(micrograph="m.mrc", x=0.0, y=0.0, R=A.T if transpose else A, score=10.0, df1_A=dfu,
                         df2_A=dfv, angast_deg=ang, image=f"{i + 1:06d}@stack.mrcs", subset=1 + i % 2))
    with mrcfile.new(d / "stack.mrcs", overwrite=True) as m:
        m.set_data(np.stack(imgs))
        m.set_image_stack()
        m.voxel_size = PIX
    write_star(rows, d / "p.star", PIX, 300.0, 2.7, 0.07, 0, N)
    return d / "p.star"


def test_reconstruction_matches_template(tmp_path):
    mod = _tool()
    star = _dataset(tmp_path, mod)
    out = tmp_path / "check"
    assert mod.main([str(star), "--template", str(tmp_path / "template.mrc"), "--out", str(out), "--pixel", str(PIX),
                     "--max-particles", "0", "--compare-raw", "--diameter-A", "60"]) == 0
    rep = json.loads((out / "check.json").read_text())
    print(rep)
    assert rep["particles"] == 300 and rep["box"] == N
    assert rep["correlation_with_template"] > 0.85                   # right orientations, positive contrast
    assert rep["correlation_with_template_raw"] < rep["correlation_with_template"] - 0.3   # unnormalised: ball
    assert abs(rep["image_mean_median"] - 30) < 1
    for f in ("map.mrc", "half1.mrc", "half2.mrc", "template.mrc", "map_raw.mrc", "check.png", "fsc.png"):
        assert (out / f).exists()


def test_wrong_rotations_do_not_match(tmp_path):
    mod = _tool()
    star = _dataset(tmp_path, mod, n=200, offset=0.0, transpose=True)
    out = tmp_path / "check"
    assert mod.main([str(star), "--template", str(tmp_path / "template.mrc"), "--out", str(out), "--pixel", str(PIX),
                     "--max-particles", "0", "--diameter-A", "60"]) == 0
    rep = json.loads((out / "check.json").read_text())
    assert rep["correlation_with_template"] < 0.7               # 0.98 with the right rotations


def test_euler_matrix_is_relion():
    from smap.export import relion_matrix
    mod = _tool()
    for ang in ((10, 50, -30), (200, 120, 45), (0, 0, 0)):
        assert np.allclose(mod.euler_matrix(*ang), relion_matrix(*ang))


def test_symmetry_expansion(tmp_path):
    """Orientations from one asymmetric unit: C1 misses most of Fourier space; --symmetry fills it."""
    from smap import io as sio
    mod = _tool()
    ops = Rotation.create_group("T").as_matrix()
    sio.write_rotations_file(ops, tmp_path / "T.txt")
    star = _dataset(tmp_path, mod, n=150, offset=0.0, ops=ops)
    args = [str(star), "--template", str(tmp_path / "template.mrc"), "--pixel", str(PIX), "--max-particles", "0",
            "--diameter-A", "60"]
    assert mod.main(args + ["--out", str(tmp_path / "c1")]) == 0
    assert mod.main(args + ["--out", str(tmp_path / "sym"), "--symmetry", str(tmp_path / "T.txt")]) == 0
    c1 = json.loads((tmp_path / "c1" / "check.json").read_text())["correlation_with_template"]
    sym = json.loads((tmp_path / "sym" / "check.json").read_text())
    print(c1, sym)
    assert sym["symmetry_operators"] == 12 and sym["correlation_with_template"] > 0.9
    assert sym["correlation_with_template"] > c1 + 0.1


def test_expand_symmetry_star(tmp_path):
    import importlib.util as iu
    from smap import io as sio
    from smap.export import read_star_particles, relion_matrix
    spec = iu.spec_from_file_location("expand_symmetry", ROOT / "tools" / "expand_symmetry.py")
    ex = iu.module_from_spec(spec)
    spec.loader.exec_module(ex)
    ops = Rotation.create_group("T").as_matrix()
    sio.write_rotations_file(ops, tmp_path / "T.txt")
    A = Rotation.random(2, random_state=1).as_matrix()
    rows = [dict(micrograph="m.mrc", x=10.0 * i, y=0.0, R=A[i], score=9.0, df1_A=9000.0, df2_A=8900.0,
                 angast_deg=5.0, image=f"{i + 1:06d}@s.mrcs", subset=1 + i) for i in range(2)]
    write_star(rows, tmp_path / "p.star", 1.0, 300.0, 2.7, 0.07, 0, 16)
    assert ex.main([str(tmp_path / "p.star"), str(tmp_path / "T.txt"), "-o", str(tmp_path / "e.star")]) == 0
    out = read_star_particles(tmp_path / "e.star")
    assert len(out) == 24
    for i in range(2):
        mine = [r for r in out if r["rlnImageName"] == f"{i + 1:06d}@s.mrcs"]
        assert len(mine) == 12 and {r["rlnRandomSubset"] for r in mine} == {str(1 + i)}
        got = [relion_matrix(float(r["rlnAngleRot"]), float(r["rlnAngleTilt"]), float(r["rlnAnglePsi"])) for r in mine]
        for M in ops:                                                  # every A M^T present once
            assert sum(np.allclose(G, A[i] @ M.T, atol=1e-5) for G in got) == 1
    assert "data_optics" in (tmp_path / "e.star").read_text()
