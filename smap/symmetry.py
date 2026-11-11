"""Point-group symmetry for the orientation search (new; not part of v2.1).

For a template with point-group symmetry G (rotations S about the template centre, model
frame: V(S x) = V(x)), the template of rotation R equals that of R Sᵀ: the central slice is
V(Rᵀk) and V(S Rᵀk) = V(Rᵀk). Its viewing direction in the model frame, d = Rᵀe_z (the third
row of R), maps to S d. Keeping only the orientations whose d lies in one cell of the group
(the d closest to a fixed generic unit vector u among its orbit {S d}) and every in-plane angle
removes the |G| − 1 duplicates of each template and leaves one asymmetric unit.

Unlike calculate_search_grid('I', ...), which needs the model in a standard orientation, this
works with the symmetry operators of the model as deposited (mmCIF pdbx_struct_oper_list), or,
for entries that deposit the full assembly with only the identity listed (many cryo-EM entries),
with the rotations that superpose the chains onto each other (``--from-chains``).

``symmetryFile`` in the .par: a rotations file (smap.io format) with the |G| rotation matrices.
Write it with ``python -m smap.symmetry MODEL.cif --out ops.txt`` (add ``--from-chains`` for a
deposited full assembly).
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
        """True if M is one of the operators (within tol)."""
        return bool((np.abs(flat - M.reshape(9)).max(axis=1) < tol).any())

    if not member(np.eye(3)):
        return False
    return all(member(a @ b) for a in S for b in S)


def unique_ops(S: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """The operators without duplicates (within tol); the first occurrence is kept."""
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
        """Rotation part (3x3) of one pdbx_struct_oper_list row."""
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


def _atom_site_ca(path) -> dict:
    """{chain: {residue: xyz}} of the C-alpha atoms (ATOM records; auth chain and residue ids)."""
    keys, out = [], {}
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("_atom_site."):
                keys.append(line.split()[0].split(".", 1)[1])
                continue
            if not line.startswith("ATOM") or not keys:
                continue
            f = line.split()
            if len(f) != len(keys):
                continue
            r = dict(zip(keys, f))
            if r.get("label_atom_id") != "CA" or r.get("label_alt_id", ".") not in (".", "?", "A"):
                continue
            ch = r.get("auth_asym_id", r.get("label_asym_id"))
            res = r.get("auth_seq_id", r.get("label_seq_id")) + r.get("pdbx_PDB_ins_code", "?").strip("?.")
            out.setdefault(ch, {})[res] = (float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"]))
    return out


def chain_operators(path, min_common: int = 20, max_rmsd: float = 1.0) -> tuple:
    """Rotations about the assembly centroid that superpose the first chain onto every chain.

    For entries that deposit the full assembly (every chain is a copy of one subunit). Chains
    with fewer than ``min_common`` C-alpha atoms in common with the first chain are skipped (other
    entities). Returns (S (n, 3, 3), rmsd per chain (A)); requires a rotation group."""
    ca = _atom_site_ca(path)
    if not ca:
        raise ValueError(f"{path}: no C-alpha atoms in _atom_site")
    chains = list(ca)
    ref = ca[chains[0]]
    c = np.array([xyz for d in ca.values() for xyz in d.values()]).mean(axis=0)
    S, rmsd = [], []
    for ch in chains:
        common = [k for k in ref if k in ca[ch]]
        if len(common) < min_common:
            continue
        P = np.array([ref[k] for k in common]) - c
        Q = np.array([ca[ch][k] for k in common]) - c
        U, _, Vt = np.linalg.svd(P.T @ Q)                     # Kabsch, rotation only: Q ~ P M^T
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        M = Vt.T @ D @ U.T
        S.append(M)
        rmsd.append(float(np.sqrt(np.mean(np.sum((P @ M.T - Q) ** 2, axis=1)))))
    S, rmsd = np.array(S), np.array(rmsd)
    if rmsd.max() > max_rmsd:
        raise ValueError(f"{path}: chain superposition rmsd up to {rmsd.max():.2f} A about the centroid "
                         "(not a symmetric assembly, or an incomplete one)")
    S = unique_ops(S)
    if not is_group(S, tol=2e-3):
        raise ValueError(f"{path}: the {len(S)} chain rotations do not form a group (incomplete assembly?)")
    return S, rmsd


def asu_mask(R: np.ndarray, S: np.ndarray, u: np.ndarray = U_REF) -> np.ndarray:
    """True for orientations whose viewing direction is the representative of its orbit."""
    d = np.asarray(R, dtype=np.float64)[:, 2, :]                 # (n, 3): Rᵀ e_z
    orbit = np.einsum("kij,nj->nki", np.asarray(S, dtype=np.float64), d)   # S_k d
    dots = orbit @ u                                            # (n, k)
    return d @ u >= dots.max(axis=1) - 1e-9


def reduce_grid(R: np.ndarray, S: np.ndarray) -> np.ndarray:
    """The orientations of R whose viewing direction is the representative of its orbit under S."""
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
    """Name of the polyhedral group of order n (T, O, I); otherwise the cyclic or dihedral candidates."""
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
    ap.add_argument("--from-chains", action="store_true",
                    help="derive the rotations by superposing the chains of a deposited full assembly "
                         "(when _pdbx_struct_oper_list holds only the identity)")
    a = ap.parse_args(argv)
    from . import io as sio
    if a.from_chains:
        S, rmsd = chain_operators(a.model)
        print(f"{len(rmsd)} chains superposed on the first: C-alpha rmsd median {np.median(rmsd):.3f} A, "
              f"max {rmsd.max():.3f} A")
    else:
        S = cif_point_operators(a.model)
    sio.write_rotations_file(S, a.out)
    print(f"{len(S)} rotations ({point_group_name(len(S))}), group checked -> {a.out}")
    if len(S) == 1 and not a.from_chains:
        print("only the identity is listed: for an entry that deposits the full assembly, use --from-chains")
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
