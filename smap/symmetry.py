"""Point-group symmetry for the orientation search (new; not part of v2.1).

For a template with point-group symmetry G (rotations S about the template centre, model
frame: V(S x) = V(x)), the template of rotation R equals that of R Sᵀ: the central slice is
V(Rᵀk) and V(S Rᵀk) = V(Rᵀk). Its viewing direction in the model frame, d = Rᵀe_z (the third
row of R), maps to S d. Keeping only the orientations whose d lies in one cell of the group
(the d closest to a fixed generic unit vector u among its orbit {S d}) and every in-plane angle
removes the |G| − 1 duplicates of each template and leaves one asymmetric unit.

Unlike calculate_search_grid('I', ...), which needs the model in a standard orientation, this
works with the symmetry operators of the model as deposited (mmCIF pdbx_struct_oper_list).

``symmetryFile`` in the .par: a rotations file (smap.io format) with the |G| rotation matrices.
Write it with ``python -m smap.symmetry MODEL.cif --out ops.txt``.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import numpy as np

U_REF = np.array([0.2311, 0.6106, 0.7574])     # generic direction (on no symmetry axis)
U_REF = U_REF / np.linalg.norm(U_REF)


def _cif_category(path, cat: str) -> list:
    """Rows of one mmCIF category (loop or single record) as dicts of strings."""
    lines = Path(path).read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_" and i + 1 < len(lines) and lines[i + 1].strip().startswith(cat + "."):
            keys = []
            i += 1
            while i < len(lines) and lines[i].strip().startswith(cat + "."):
                keys.append(lines[i].strip().split(".", 1)[1].split()[0])
                i += 1
            toks = []
            while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(("_", "loop_", "#")):
                toks += shlex.split(lines[i])
                i += 1
            return [dict(zip(keys, toks[k:k + len(keys)])) for k in range(0, len(toks) - len(keys) + 1, len(keys))]
        i += 1
    rec = {}
    for ln in lines:
        s = ln.strip()
        if s.startswith(cat + "."):
            parts = shlex.split(s)
            if len(parts) >= 2:
                rec[parts[0].split(".", 1)[1]] = parts[1]
    return [rec] if rec else []


def is_group(S: np.ndarray, tol: float = 1e-3) -> bool:
    """Closure under products and presence of the identity (rotation matrices)."""
    S = np.asarray(S, dtype=np.float64)
    flat = S.reshape(len(S), 9)

    def member(M):
        return bool((np.abs(flat - M.reshape(9)).max(axis=1) < tol).any())

    if not member(np.eye(3)):
        return False
    return all(member(a @ b) for a in S for b in S)


def unique_ops(S: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    out = []
    for M in S:
        if not any(np.abs(M - N).max() < tol for N in out):
            out.append(M)
    return np.array(out)


def cif_point_operators(path) -> np.ndarray:
    """Rotation parts (n, 3, 3) of the point-symmetry operators in pdbx_struct_oper_list.

    Uses the operators typed 'identity operation' / 'point symmetry operation' when present
    (virus entries also list frame transformations), otherwise all; requires a group."""
    recs = _cif_category(path, "_pdbx_struct_oper_list")
    if not recs:
        raise ValueError(f"{path}: no _pdbx_struct_oper_list")

    def mat(r):
        return np.array([[float(r[f"matrix[{i}][{j}]"]) for j in (1, 2, 3)] for i in (1, 2, 3)])

    typed = [r for r in recs if r.get("type", "").lower() in ("identity operation", "point symmetry operation")]
    for subset in (typed, recs):
        if not subset:
            continue
        S = unique_ops(np.array([mat(r) for r in subset]))
        dets = np.linalg.det(S)
        if np.allclose(dets, 1.0, atol=1e-2) and is_group(S, tol=2e-3):
            return S
    raise ValueError(f"{path}: the operators do not form a rotation group")


def asu_mask(R: np.ndarray, S: np.ndarray, u: np.ndarray = U_REF) -> np.ndarray:
    """True for orientations whose viewing direction is the representative of its orbit."""
    d = np.asarray(R, dtype=np.float64)[:, 2, :]                 # (n, 3): Rᵀ e_z
    orbit = np.einsum("kij,nj->nki", np.asarray(S, dtype=np.float64), d)   # S_k d
    dots = orbit @ u                                            # (n, k)
    return d @ u >= dots.max(axis=1) - 1e-9


def reduce_grid(R: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.asarray(R)[asu_mask(R, S)]


def check_assembly(S: np.ndarray, xyz: np.ndarray, n_sample: int = 2000, seed: int = 0) -> np.ndarray:
    """Distance (A) from each operator image of sampled atoms to the nearest atom of the model.

    xyz: (n, 3) coordinates of the full assembly (the template). The operators act about the
    atom centroid, which is the symmetry centre of a complete assembly. Returns (|G|, 2):
    median and 95th percentile per operator (~0 for a symmetric model and correct operators)."""
    from scipy.spatial import cKDTree
    xyz = np.asarray(xyz, dtype=np.float64)
    c = xyz.mean(axis=0)
    rng = np.random.default_rng(seed)
    pts = xyz[rng.choice(len(xyz), min(n_sample, len(xyz)), replace=False)] - c
    tree = cKDTree(xyz - c)
    out = []
    for M in np.asarray(S, dtype=np.float64):
        d, _ = tree.query(pts @ M.T)
        out.append((np.median(d), np.percentile(d, 95)))
    return np.array(out)


def point_group_name(n: int) -> str:
    return {12: "T", 24: "O", 60: "I"}.get(n, f"C{n} or D{n // 2}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write the point-symmetry rotations of an mmCIF entry "
                                             "for the .par key symmetryFile")
    ap.add_argument("model", help="mmCIF with _pdbx_struct_oper_list (e.g. the asymmetric-unit entry)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--assembly", default=None,
                    help="template model (full assembly, .cif/.pdb): check that every operator maps "
                         "its atoms onto atoms (median distance < --tol A)")
    ap.add_argument("--tol", type=float, default=0.5)
    a = ap.parse_args(argv)
    from . import io as sio
    S = cif_point_operators(a.model)
    sio.write_rotations_file(S, a.out)
    print(f"{len(S)} rotations ({point_group_name(len(S))}), group checked -> {a.out}")
    if a.assembly:
        from .structure import read_cif_file, read_pdb_file
        rd = read_cif_file if Path(a.assembly).suffix == ".cif" else read_pdb_file
        xyz = rd(a.assembly).xyz.T
        d = check_assembly(sio.read_rotations_file(a.out), xyz)
        worst = int(np.argmax(d[:, 0]))
        print(f"assembly check ({len(xyz)} atoms): median distance max {d[:, 0].max():.3f} A "
              f"(operator {worst + 1}), 95th percentile max {d[:, 1].max():.3f} A")
        if d[:, 0].max() > a.tol:
            print("FAILED: the operators do not map the template onto itself "
                  "(incomplete assembly, or operators of another frame)")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
