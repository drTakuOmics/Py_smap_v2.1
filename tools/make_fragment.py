"""Spherical fragments of an atomic model as search templates (new; not part of v2.1).

Two selections, both centred on the mass-weighted centre of the whole model (as in Rickgauer
et al., Mol Cell 2024, for the LSU model 6ek0):

* ``--radius R`` (nm): atoms within R of the centre, optionally restricted to ``--chains``.
  The 700 kDa LSU template of that paper = 28S rRNA (6ek0 chain L5) within 8.5 nm.
* ``--mw M`` (kDa): all atoms (or ``--chains``) sorted by distance from the centre, the radius
  expanded until the fragment reaches M (the paper's 200-1000 kDa LSU series).

The selected atom lines are copied verbatim from the input mmCIF (same columns and B-factors,
so the SMAP reader sees exactly the original format). Mass: heavy atoms as in the file (this
reproduces the paper's 200 and 1000 kDa radii, 4.34 and 7.84 nm, within 1%); ``--implicit-h``
adds hydrogens by residue class instead (H/heavy mass ~0.078 protein, ~0.035 nucleic acid).
``mw_smap_kDa`` is the mass of the ATOM records only: SMAP v2.1 (and the port) ignore HETATM
records (modified nucleotides, ions), so that is the template the search uses.

usage:
    python tools/make_fragment.py MODEL.cif --radius 8.5 --chains L5 --out frag_28S_r85.cif
    python tools/make_fragment.py MODEL.cif --mw 300 --out frag_300kDa.cif
    python tools/make_fragment.py MODEL.cif --mw 200 300 400 600 800 1000 --out-dir fragments/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "P": 30.974, "S": 32.06, "SE": 78.97,
        "MG": 24.305, "ZN": 65.38, "K": 39.098, "NA": 22.990, "CL": 35.45, "CA": 40.078, "FE": 55.845}
NUC = {"A", "C", "G", "U", "DA", "DC", "DG", "DT", "I", "PSU", "OMG", "OMC", "OMU", "A2M", "2MG", "M2G",
       "5MC", "1MA", "7MG", "B9B", "UR3", "4AC", "MA6", "6MZ", "B8N", "E7G"}
H_FRAC = {"protein": 0.078, "nucleic": 0.035}


def read_atoms(path):
    """(header_lines, atom_lines, table) for the single atom_site loop of an mmCIF file."""
    header, atoms, cols = [], [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                atoms.append(line)
            elif not atoms:
                header.append(line)
                if line.startswith("_atom_site."):
                    cols.append(line.strip().split(".", 1)[1])
    idx = {c: i for i, c in enumerate(cols)}
    f = [a.split() for a in atoms]
    tab = dict(
        xyz=np.array([[float(r[idx["Cartn_x"]]), float(r[idx["Cartn_y"]]), float(r[idx["Cartn_z"]])] for r in f]),
        element=np.array([r[idx["type_symbol"]].upper() for r in f]),
        chain=np.array([r[idx["auth_asym_id"] if "auth_asym_id" in idx else idx["label_asym_id"]] for r in f]),
        comp=np.array([r[idx["label_comp_id"]] for r in f]),
    )
    return header, atoms, tab


def masses(tab):
    """Mass of every atom as modelled, and with hydrogens: when the model has none, a fraction
    per residue class (protein, nucleic acid) is added. Returns (heavy, with_h)."""
    heavy = np.array([MASS.get(e, 12.0) for e in tab["element"]])
    heavy[tab["element"] == "H"] = MASS["H"]
    has_h = (tab["element"] == "H").any()
    nuc = np.isin(tab["comp"], list(NUC))
    with_h = heavy if has_h else heavy * np.where(nuc, 1 + H_FRAC["nucleic"], 1 + H_FRAC["protein"])
    return heavy, with_h


def fragment(path, radius_nm=None, mw_kda=None, chains=None, implicit_h=False):
    """Selected atom lines and a summary; centre = heavy-atom COM of the whole model."""
    header, atoms, tab = read_atoms(path)
    heavy, with_h = masses(tab)
    mass = with_h if implicit_h else heavy
    is_atom = np.array([a.startswith("ATOM") for a in atoms])
    com = (tab["xyz"] * heavy[:, None]).sum(0) / heavy.sum()
    d = np.linalg.norm(tab["xyz"] - com, axis=1)
    pool = np.isin(tab["chain"], list(chains)) if chains else np.ones(len(d), bool)
    if radius_nm is not None:
        sel = np.nonzero(pool & (d <= radius_nm * 10.0))[0]
        r = radius_nm
    else:
        order = np.nonzero(pool)[0][np.argsort(d[pool], kind="stable")]
        cum = np.cumsum(mass[order]) / 1000.0
        n = int(np.searchsorted(cum, mw_kda)) + 1
        sel = np.sort(order[:n])
        r = float(d[order[n - 1]] / 10.0)
    info = dict(model=str(path), centre_A=[round(float(c), 3) for c in com], radius_nm=round(float(r), 3),
                chains=sorted(set(tab["chain"][sel])), restricted_to=sorted(chains) if chains else None,
                n_atoms=int(sel.size), n_atoms_smap=int(is_atom[sel].sum()),
                mass_convention="implicit H" if implicit_h else "heavy atoms",
                mw_kDa=round(float(mass[sel].sum() / 1000), 1),
                mw_smap_kDa=round(float(mass[sel][is_atom[sel]].sum() / 1000), 1),
                mw_with_h_kDa=round(float(with_h[sel].sum() / 1000), 1),
                model_mw_kDa=round(float(mass.sum() / 1000), 1))
    return header, [atoms[i] for i in sel], info


def write(header, lines, out, info):
    """Write the fragment as mmCIF (header and atom lines) and its description as .json."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.writelines(header)
        fh.writelines(lines)
        fh.write("#\n")
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Spherical fragment of an atomic model (mmCIF) around its COM")
    ap.add_argument("model")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--radius", type=float, help="nm")
    g.add_argument("--mw", type=float, nargs="+", help="target MW(s) in kDa")
    ap.add_argument("--chains", nargs="+", help="restrict to these auth chain IDs (e.g. L5 = 28S rRNA in 6ek0)")
    ap.add_argument("--implicit-h", action="store_true", help="count implicit hydrogens in the MW")
    ap.add_argument("--out", help="output .cif (single fragment)")
    ap.add_argument("--out-dir", help="output folder (one file per --mw value)")
    a = ap.parse_args(argv)
    jobs = [(None, m) for m in a.mw] if a.mw else [(a.radius, None)]
    for rad, mw in jobs:
        header, lines, info = fragment(a.model, rad, mw, a.chains, a.implicit_h)
        out = a.out if (a.out and len(jobs) == 1) else Path(a.out_dir or ".") / (
            f"{Path(a.model).stem}_{'r%.1fnm' % rad if rad else '%gkDa' % mw}.cif")
        write(header, lines, out, info)
        print(f"{out}: {info['n_atoms']} atoms, {info['mw_kDa']} kDa (read by SMAP {info['mw_smap_kDa']}), "
              f"r = {info['radius_nm']} nm, chains {','.join(info['chains'][:8])}{'...' if len(info['chains']) > 8 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
