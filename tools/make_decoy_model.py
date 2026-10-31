#!/usr/bin/env python3
"""Build a mass- and shape-matched decoy model for a 2DTM target-decoy run.

A decoy for 2DTM must be a *different* structure (a mirror image is not a decoy: its
projections equal the target's). It should match the searched template in signal (atom
count, i.e. SMAP mass = atoms x 13.824 Da) and roughly in size/shape, so that its score
distribution on the same images estimates the false-positive tail of the target search.
A structurally related decoy (e.g. dynein motor for RNF213, which shares the AAA+ ring
architecture) is a stringent ("hard") decoy: it can partly match true particles and
therefore tends to over-estimate false positives.

What it does:
  1. reads the decoy mmCIF (model 1, ATOM records, no hydrogens; optional --chains /
     --exclude residue ranges) and the target model actually searched (--match, e.g. the
     anchor CIF; same filters via --match-chains / --match-exclude), or --match-atoms N;
  2. trims the decoy with whole residues to the target's atom count:
       ellipsoid (default): decoy and target principal axes are paired by size and residues
         are kept in order of their distance in the target's principal-axis metric, so the
         decoy keeps the target's overall proportions;
       sphere: residues closest to the decoy centre of mass (like rnf213_fragment_pilot.py);
       none: no trimming (report only);
  3. writes the kept ATOM lines unchanged (same column layout as RCSB files, which the SMAP
     CIF reader expects) and a JSON/text report comparing atoms, SMAP mass, atoms/residue
     (side-chain completeness), radius of gyration and principal-axis spans.

B-factors in the file do not matter for SMAP's potential when bArb is 0 (default).

usage:
  python make_decoy_model.py --decoy 8DYU.cif --match anchor_noE3_pilot/<anchor>.cif \\
         --out decoy/8DYU_matched [--chains A,B] [--exclude A:1-100] [--trim ellipsoid]
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import numpy as np

SMAP_DA_PER_ATOM = 13.824


def parse_ranges(spec):
    out = []
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        ch, rng = item.split(":")
        a, b = rng.split("-")
        out.append((ch, int(a), int(b)))
    return out


def read_atoms(path, chains=None, exclude=None):
    """mmCIF _atom_site (model 1, ATOM, no H). Returns header lines, kept lines, arrays."""
    lines = Path(path).read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_" and i + 1 < len(lines) and lines[i + 1].startswith("_atom_site."):
            break
        i += 1
    if i == len(lines):
        raise SystemExit(f"{path}: no _atom_site loop")
    hdr = []
    j = i + 1
    while lines[j].startswith("_atom_site."):
        hdr.append(lines[j].strip())
        j += 1
    col = {h.split(".", 1)[1]: k for k, h in enumerate(hdr)}
    need = ["group_PDB", "type_symbol", "Cartn_x", "Cartn_y", "Cartn_z", "auth_asym_id", "auth_seq_id"]
    miss = [n for n in need if n not in col]
    if miss:
        raise SystemExit(f"{path}: atom_site lacks {miss}")
    exclude = exclude or []
    keep_lines, xyz, chain, resi, elem = [], [], [], [], []
    for line in lines[j:]:
        s = line.strip()
        if not s or s.startswith(("#", "loop_", "_", "data_")):
            if s.startswith(("loop_", "data_")):
                break
            continue
        f = shlex.split(s) if ("'" in s or '"' in s) else s.split()
        if len(f) != len(hdr) or f[col["group_PDB"]] != "ATOM":
            continue
        if "pdbx_PDB_model_num" in col and f[col["pdbx_PDB_model_num"]] not in ("1", "?"):
            continue
        el = f[col["type_symbol"]].upper()
        if el in ("H", "D"):
            continue
        ch, rn = f[col["auth_asym_id"]], int(f[col["auth_seq_id"]])
        if chains and ch not in chains:
            continue
        if any(ch == c and a <= rn <= b for c, a, b in exclude):
            continue
        keep_lines.append(line)
        xyz.append([float(f[col["Cartn_x"]]), float(f[col["Cartn_y"]]), float(f[col["Cartn_z"]])])
        chain.append(ch)
        resi.append(rn)
        elem.append(el)
    if not xyz:
        raise SystemExit(f"{path}: no atoms after filtering")
    return lines[i:j], keep_lines, np.array(xyz), np.array(chain), np.array(resi), elem


def shape(xyz):
    c = xyz - xyz.mean(0)
    w, v = np.linalg.eigh(np.cov(c.T))
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    proj = c @ v
    return dict(rg_A=float(np.sqrt((c ** 2).sum(1).mean())),
                pc_sigma_A=[float(x) for x in np.sqrt(w)],
                pc_span_A=[float(x) for x in proj.max(0) - proj.min(0)]), v, np.sqrt(w)


def describe(xyz, chain, resi, elem):
    nres = len(set(zip(chain.tolist(), resi.tolist())))
    s, _, _ = shape(xyz)
    return dict(atoms=int(len(xyz)), residues=int(nres), atoms_per_residue=round(len(xyz) / max(1, nres), 2),
                smap_mass_kDa=round(len(xyz) * SMAP_DA_PER_ATOM / 1000, 1),
                chains=sorted(set(chain.tolist())), elements={e: elem.count(e) for e in sorted(set(elem))}, **s)


def trim(xyz, chain, resi, n_target, mode, target_sigma=None):
    keys = list(zip(chain.tolist(), resi.tolist()))
    uniq = list(dict.fromkeys(keys))
    idx = {k: i for i, k in enumerate(uniq)}
    rid = np.array([idx[k] for k in keys])
    cen = np.zeros((len(uniq), 3))
    np.add.at(cen, rid, xyz)
    cnt = np.bincount(rid, minlength=len(uniq))
    cen /= cnt[:, None]
    com = xyz.mean(0)
    if mode == "sphere":
        d = np.linalg.norm(cen - com, axis=1)
    else:                                   # ellipsoid in the target's proportions
        _, v, _ = shape(xyz)
        u = (cen - com) @ v                 # decoy principal-axis frame (largest first)
        d = np.sqrt(((u / np.asarray(target_sigma)) ** 2).sum(1))
    order = np.argsort(d)
    keep_res, total = [], 0
    for r in order:
        if total >= n_target:
            break
        keep_res.append(r)
        total += cnt[r]
    keep = np.isin(rid, keep_res)
    return keep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decoy", required=True, help="decoy mmCIF (e.g. 8DYU.cif)")
    ap.add_argument("--chains", default=None, help="decoy auth chain IDs to use, e.g. A,B")
    ap.add_argument("--exclude", default=None, help="decoy residues to drop, e.g. A:1-1200,B:1-50")
    ap.add_argument("--match", default=None, help="target model actually searched (mmCIF)")
    ap.add_argument("--match-chains", default=None)
    ap.add_argument("--match-exclude", default=None, help="e.g. A:3589-4926 (E3 omitted)")
    ap.add_argument("--match-atoms", type=int, default=None, help="target atom count instead of --match")
    ap.add_argument("--trim", choices=("ellipsoid", "sphere", "none"), default="ellipsoid")
    ap.add_argument("--out", required=True, help="output stem (writes STEM.cif, STEM.json, STEM.txt)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if any(out.with_suffix(s).exists() for s in (".cif", ".json")):
        raise SystemExit(f"{out}.cif/.json exist; choose a new --out")

    hdr, lines, xyz, chain, resi, elem = read_atoms(
        a.decoy, a.chains.split(",") if a.chains else None, parse_ranges(a.exclude))
    rep = dict(decoy=str(a.decoy), decoy_chains=a.chains, decoy_exclude=a.exclude,
               decoy_before=describe(xyz, chain, resi, elem), trim=a.trim)
    tsig = None
    if a.match:
        _, _, txyz, tch, tres, tel = read_atoms(
            a.match, a.match_chains.split(",") if a.match_chains else None, parse_ranges(a.match_exclude))
        rep["target"] = dict(model=str(a.match), chain_filter=a.match_chains, exclude=a.match_exclude,
                             **describe(txyz, tch, tres, tel))
        n_target = len(txyz)
        tsig = shape(txyz)[2]
    elif a.match_atoms:
        n_target = a.match_atoms
    else:
        n_target = len(xyz)
    if a.trim == "ellipsoid" and tsig is None:
        tsig = shape(xyz)[2]
    keep = np.ones(len(xyz), bool) if a.trim == "none" or n_target >= len(xyz) else \
        trim(xyz, chain, resi, n_target, a.trim, tsig)
    kept_lines = [ln for ln, k in zip(lines, keep) if k]
    rep["decoy_after"] = describe(xyz[keep], chain[keep], resi[keep], [e for e, k in zip(elem, keep) if k])
    rep["atom_ratio_decoy_over_target"] = round(int(keep.sum()) / n_target, 3)
    warn = []
    if rep["decoy_after"]["atoms_per_residue"] < 7.0:
        warn.append("decoy has < 7 atoms/residue: side chains truncated in the model; its potential is "
                    "weaker than a complete model of the same residue count")
    if "target" in rep and abs(rep["target"]["atoms_per_residue"] - rep["decoy_after"]["atoms_per_residue"]) > 1.0:
        warn.append("atoms/residue differ by > 1 between target and decoy models (different completeness)")
    if n_target > len(xyz):
        warn.append(f"decoy has fewer atoms ({len(xyz)}) than the target ({n_target}); not mass matched")
    rep["warnings"] = warn

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".cif"), "w") as fh:
        fh.write(f"data_{out.name}\n#\n")
        fh.write("\n".join(hdr) + "\n")
        fh.write("\n".join(kept_lines) + "\n#\n")
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    t = rep.get("target")
    d0, d1 = rep["decoy_before"], rep["decoy_after"]
    rows = [("", "target" if t else "-", "decoy (input)", "decoy (written)")]
    for k in ("atoms", "residues", "atoms_per_residue", "smap_mass_kDa", "rg_A"):
        rows.append((k, t[k] if t else "", d0[k], d1[k]))
    rows.append(("pc_span_A", [round(x) for x in t["pc_span_A"]] if t else "",
                 [round(x) for x in d0["pc_span_A"]], [round(x) for x in d1["pc_span_A"]]))
    text = "\n".join(f"{r[0]:18s} {str(r[1]):>20s} {str(r[2]):>20s} {str(r[3]):>20s}" for r in rows)
    text += "\n" + "\n".join("WARNING: " + w for w in warn) + "\n"
    out.with_suffix(".txt").write_text(text)
    print(text)
    print(f"wrote {out.with_suffix('.cif')} ({int(keep.sum())} atoms), {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
