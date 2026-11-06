"""Point-group reduction of the search grid (smap.symmetry; new, not part of v2.1)."""

from __future__ import annotations

import numpy as np
import pytest

from smap import io as sio
from smap.grid import calculate_search_grid
from smap.structure import Atoms, electrostatic_potential
from smap.symmetry import asu_mask, check_assembly, cif_point_operators, is_group, reduce_grid
from smap.utils import crop_or_pad


def _rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.deg2rad(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


AXIS = np.array([0.3, -0.5, 0.81])
C4 = np.array([_rot(AXIS, 90 * k) for k in range(4)])


def _angle_deg(A, B):
    c = (np.trace(A.T @ B) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def test_c4_group_and_grid_fraction():
    assert is_group(C4)
    assert not is_group(C4[:3])
    R, _ = calculate_search_grid("C1", 10, 10)
    keep = asu_mask(R, C4)
    assert abs(keep.mean() - 1 / 4) < 0.02


def test_one_member_of_each_orbit_kept():
    rng = np.random.default_rng(1)
    for _ in range(50):
        q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        R0 = q * np.sign(np.linalg.det(q))
        orbit = np.array([R0 @ S.T for S in C4])
        assert asu_mask(orbit, C4).sum() == 1


def test_reduced_grid_covers_full_grid():
    """Every orientation of the full grid has a symmetry mate within one grid step."""
    R, _ = calculate_search_grid("C1", 10, 10)
    Rk = reduce_grid(R, C4)
    rng = np.random.default_rng(2)
    for i in rng.choice(len(R), 40, replace=False):
        best = min(_angle_deg(R[i] @ S.T, B) for S in C4 for B in Rk)
        assert best < 10.0


def _c4_volume(n=64, a=2.0):
    """C4-symmetric atom set about a tilted axis through an off-origin centre, as a volume
    in the port's [row=y, col=x, page=z] layout (atom mean at the centre voxel)."""
    rng = np.random.default_rng(3)
    base = rng.normal(0, 7, (40, 3)) + np.array([12.0, 4.0, -3.0])
    c = np.array([30.0, -20.0, 15.0])
    xyz = np.concatenate([base @ S.T for S in C4]) + c          # c + S x
    atoms = Atoms(xyz.T.copy(), ["C"] * len(xyz), np.zeros(len(xyz)), ["A"] * len(xyz))
    V = electrostatic_potential(atoms, a, b_arb=20.0)
    return crop_or_pad(V, [n] * 3, 0).astype(np.float32)


def test_templates_equal_for_symmetry_mates(tmp_path):
    from smap.search_global import make_templates, prepare
    sio.mw(_c4_volume(), tmp_path / "vol.mrc", 2.0)
    sio.mw(np.random.default_rng(0).standard_normal((96, 96)).astype(np.float32),
           tmp_path / "img.mrc", 2.0)
    ops = tmp_path / "ops.txt"
    sio.write_rotations_file(C4, ops)
    par = tmp_path / "s.par"
    par.write_text("function search_global\nnCores 1\n"
                   f"imageFile {tmp_path / 'img.mrc'}\nstructureFile {tmp_path / 'vol.mrc'}\n"
                   f"outputDir {tmp_path / 'out'}\naPerPix 2.0\ndefocus 15000.0 14000.0 30.0\n"
                   f"angle_inc 20\nT_sample 0\narbThr 4.0\nsymmetryFile {ops}\n")
    st = prepare(par, write_files=False)
    n_full = calculate_search_grid("C1", 20, 20)[0].shape[0]
    assert abs(st.R_full.shape[0] / n_full - 1 / 4) < 0.05
    assert any("symmetry: 4 operators" in m for m in st.log_lines)

    R = _rot([0.2, 0.9, -0.3], 37.0)
    S = sio.read_rotations_file(ops)                           # 4-decimal file precision
    st.R_full = np.stack([R, R @ S[1].T, R @ S[2].T, R @ _rot([1, 0, 0], 25.0)])
    t = make_templates(st, [1, 2, 3, 4]).reshape(4, -1)
    t = t - t.mean(axis=1, keepdims=True)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    ncc = t @ t[0]
    assert ncc[1] > 0.99 and ncc[2] > 0.99
    assert ncc[3] < 0.9


CIF = """data_test
loop_
_pdbx_struct_oper_list.id
_pdbx_struct_oper_list.type
_pdbx_struct_oper_list.name
_pdbx_struct_oper_list.symmetry_operation
_pdbx_struct_oper_list.matrix[1][1]
_pdbx_struct_oper_list.matrix[1][2]
_pdbx_struct_oper_list.matrix[1][3]
_pdbx_struct_oper_list.vector[1]
_pdbx_struct_oper_list.matrix[2][1]
_pdbx_struct_oper_list.matrix[2][2]
_pdbx_struct_oper_list.matrix[2][3]
_pdbx_struct_oper_list.vector[2]
_pdbx_struct_oper_list.matrix[3][1]
_pdbx_struct_oper_list.matrix[3][2]
_pdbx_struct_oper_list.matrix[3][3]
_pdbx_struct_oper_list.vector[3]
{rows}
#
"""


def _cif_rows(ops, types):
    rows = []
    for i, (M, ty) in enumerate(zip(ops, types), start=1):
        v = " ".join(f"{M[r, 0]:.6f} {M[r, 1]:.6f} {M[r, 2]:.6f} {5.0 * r:.3f}" for r in range(3))
        rows.append(f"{i} '{ty}' ? ? {v}")
    return "\n".join(rows)


def test_cif_point_operators(tmp_path):
    P = _rot([1, 2, 3], 17.0)                                  # 'transform to point frame'
    types = ["identity operation"] + ["point symmetry operation"] * 3 + ["transform to point frame"]
    f = tmp_path / "x.cif"
    f.write_text(CIF.format(rows=_cif_rows(list(C4) + [P], types)))
    S = cif_point_operators(f)
    assert S.shape == (4, 3, 3) and is_group(S)
    f2 = tmp_path / "bad.cif"
    f2.write_text(CIF.format(rows=_cif_rows(list(C4[:3]), ["point symmetry operation"] * 3)))
    with pytest.raises(ValueError):
        cif_point_operators(f2)


def test_check_assembly():
    rng = np.random.default_rng(4)
    base = rng.normal(0, 7, (300, 3)) + np.array([12.0, 4.0, -3.0])
    xyz = np.concatenate([base @ S.T for S in C4]) + np.array([30.0, -20.0, 15.0])
    S = C4.astype(np.float32).astype(np.float64)                 # file precision
    assert check_assembly(S, xyz)[:, 0].max() < 0.01
    assert check_assembly([_rot(AXIS, 45.0)], xyz)[0, 0] > 1.0
    assert check_assembly(C4, xyz[:600])[1:, 0].min() > 1.0     # incomplete assembly
