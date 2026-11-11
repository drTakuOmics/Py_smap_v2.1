#!/usr/bin/env python
"""Symmetry expansion of a RELION particle STAR file with the search's own symmetry operators.

    python tools/expand_symmetry.py EXPORT/particles.star OPS.txt -o EXPORT/particles_expanded.star

A search with a symmetryFile reports every orientation inside one asymmetric unit: all viewing
directions fall in 1/|G| of the sphere, the images are slices near one plane of Fourier space, and a
C1 reconstruction leaves the cone around the common viewing direction empty (streaks along it).
Writing each particle |G| times, at the equivalent poses A Mᵀ (A = RELION Euler_angles2matrix of the
row, M = each rotation of OPS.txt, the smap.io rotations file of the search), fills it. A C1
reconstruction of the expanded file is the symmetry-averaged map in the template's frame, so the
reconstruction program's own convention for the symmetry axes is not used. Every other column (image, CTF, origin, rlnRandomSubset: copies stay in their half-set) is kept;
the optics table is copied unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # repo root (smap)


def expand_text(text: str, ops: np.ndarray) -> tuple[str, int, int]:
    """(expanded STAR text, particles in, rows out); the particle loop is the one with rlnAngleRot."""
    from smap.export import relion_angles, relion_matrix
    lines = text.splitlines()
    out, labels, in_loop, n_in, n_out = [], [], False, 0, 0
    for ln in lines:
        s = ln.strip()
        if s.startswith("data_") or s == "loop_":
            in_loop, labels = s == "loop_", []
            out.append(ln)
            continue
        if in_loop and s.startswith("_"):
            labels.append(s.split()[0][1:])
            out.append(ln)
            continue
        f = s.split()
        if in_loop and "rlnAngleRot" in labels and len(f) == len(labels):
            i_rot, i_tilt, i_psi = (labels.index(k) for k in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"))
            A = relion_matrix(float(f[i_rot]), float(f[i_tilt]), float(f[i_psi]))
            n_in += 1
            for M in ops:
                rot, tilt, psi = relion_angles(A @ M.T)
                g = list(f)
                g[i_rot], g[i_tilt], g[i_psi] = f"{rot:.6f}", f"{tilt:.6f}", f"{psi:.6f}"
                out.append(" ".join(g))
                n_out += 1
            continue
        out.append(ln)
    return "\n".join(out) + "\n", n_in, n_out


def main(argv=None) -> int:
    from smap import io as sio
    from smap.symmetry import is_group, point_group_name
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("star")
    ap.add_argument("ops", help="rotations file (smap.io format), e.g. the search's symmetryFile")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    ops = sio.read_rotations_file(a.ops)
    if not is_group(ops, tol=2e-3):
        raise SystemExit(f"{a.ops}: the {len(ops)} rotations do not form a group")
    text, n_in, n_out = expand_text(Path(a.star).read_text(), ops)
    if not n_in:
        raise SystemExit(f"{a.star}: no particle rows with rlnAngleRot/Tilt/Psi")
    Path(a.out).write_text(text)
    print(f"{n_in} particles x {len(ops)} operators ({point_group_name(len(ops))}) = {n_out} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
