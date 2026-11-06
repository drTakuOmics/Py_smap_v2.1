#!/usr/bin/env python
"""Ribosome control with the mouse large subunit, as the authors searched it (6SWA without Ebp1).

6EK0 (used so far) is the human ribosome; the cells are mouse. The authors' template was the mouse
60S from 6SWA without Ebp1 (cisTEM database: "6swa_simulate"). This keeps the polymer chains of the
large subunit, drops everything whose entity name says 40S / 18S / Ebp1 (PA2G4) / tRNA / mRNA /
translation factors, prints the decision for every entity, writes the trimmed model, its
potential, and the same per-tile LSU searches as before with only the template changed.

  wget -O $WORK/models/6swa.cif.gz https://files.rcsb.org/download/6SWA.cif.gz && gunzip -f $WORK/models/6swa.cif.gz
  cd $PYSMAP
  python examples/decolace_tric/make_lsu_mouse.py --dest $WORK --model $WORK/models/6swa.cif
Writes models/6swa_LSU.cif, models/LSU_6SWA_SP.mrc, pars_LSU_mouse/, jobs_LSU_mouse.txt (-> results_LSU_mouse/).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shlex
import sys
from pathlib import Path

import numpy as np

EXCLUDE = (r"40S|18S|Proliferation-associated|PA2G4|Ebp1|ErbB3-binding|tRNA|transfer RNA|messenger|mRNA|"
           r"initiation factor|elongation factor|\beIF|\beEF|beta-2-like|RACK1|Receptor of activated|"
           r"SERPINE1|Hyaluronan|nascent|signal recognition")


def entities(text: str) -> dict:
    """entity id -> (type, description) from the _entity loop (or single record)."""
    lines = text.splitlines()
    out = {}
    for i, ln in enumerate(lines):
        if ln.strip() == "loop_" and i + 1 < len(lines) and lines[i + 1].startswith("_entity."):
            keys, j = [], i + 1
            while lines[j].startswith("_entity."):
                keys.append(lines[j].split(".", 1)[1].strip())
                j += 1
            toks = []
            while j < len(lines) and not lines[j].startswith(("_", "loop_", "#")):
                s = lines[j]
                if s.startswith(";"):                      # multi-line text field
                    k, buf = j + 1, [s[1:]]
                    while not lines[k].startswith(";"):
                        buf.append(lines[k])
                        k += 1
                    toks.append(" ".join(b.strip() for b in buf).strip())
                    j = k + 1
                    continue
                try:
                    toks += shlex.split(s)
                except ValueError:
                    toks += s.split()
                j += 1
            n = len(keys)
            for r in range(0, len(toks) - n + 1, n):
                d = dict(zip(keys, toks[r:r + n]))
                out[d["id"]] = (d.get("type", "?"), d.get("pdbx_description", "?"))
            return out
    rec = {}
    for ln in lines:
        if ln.startswith("_entity."):
            p = ln.split(None, 1)
            if len(p) == 2:
                rec[p[0].split(".", 1)[1]] = p[1].strip().strip("'\"")
    if rec:
        out[rec.get("id", "1")] = (rec.get("type", "?"), rec.get("pdbx_description", "?"))
    return out


def main(argv=None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (smap)
    from smap import io as sio
    from smap.structure import read_cif_file
    kit = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("prep", kit / "decolace_prep.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="LSU_6SWA")
    ap.add_argument("--tiles", nargs="+", default=["00463", "00520"])
    ap.add_argument("--exclude", default=EXCLUDE, help="regex on entity names to drop")
    a = ap.parse_args(argv)
    d = Path(a.dest).expanduser().resolve()
    src = Path(a.model).expanduser().resolve()
    text = src.read_text(errors="replace")
    ents = entities(text)

    # atom_site header -> column indices
    lines = text.splitlines()
    head = [ln.strip() for ln in lines if ln.startswith("_atom_site.")]
    cols = [h.split(".", 1)[1].split()[0] for h in head]
    ie, ic = cols.index("label_entity_id"), cols.index("auth_asym_id")
    chains: dict = {}
    for ln in lines:
        if ln.startswith("ATOM"):
            f = ln.split()
            chains.setdefault(f[ie], set()).add(f[ic])
    keep_e = set()
    print(f"{'entity':>6s}  {'keep':4s}  {'chains':>6s}  description")
    for e, (typ, desc) in sorted(ents.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        if e not in chains:
            continue                                         # no ATOM records (ligands, water)
        keep = typ == "polymer" and not re.search(a.exclude, desc, re.I)
        if keep:
            keep_e.add(e)
        print(f"{e:>6s}  {'yes' if keep else 'NO':4s}  {len(chains[e]):6d}  {desc[:90]}")
    out_cif = d / "models" / (src.stem.lower() + "_LSU.cif")
    n = 0
    with open(out_cif, "w") as fh:
        fh.write(f"data_{src.stem}_LSU\n#\nloop_\n" + "\n".join(head) + "\n")
        for ln in lines:
            if ln.startswith("ATOM") and ln.split()[ie] in keep_e:
                fh.write(ln + "\n")
                n += 1
        fh.write("#\n")
    at = read_cif_file(out_cif)
    ext = np.ptp(at.xyz, axis=1)
    print(f"kept {len(keep_e)} entities, {n:,} atoms (read back {at.xyz.shape[1]:,}), extent "
          f"{ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} A -> {out_cif}")

    src_dir = d / "pars_LSU_control"
    pars = [sorted(src_dir.glob(f"*_{k}_*.par")) for k in a.tiles]
    pars = [p[0] for p in pars if p]
    if not pars:
        raise SystemExit(f"no LSU control .par for {a.tiles} in {src_dir}")
    p0, _ = sio.read_params_file(pars[0])
    sp = prep.make_sp(kit / "tric_decolace.par", out_cif, a.name, float(p0["aPerPix"]), d / "models")
    v_new = sio.mr(sp).astype(np.float64)
    v_old = sio.mr(p0["structureFile"]).astype(np.float64)
    print(f"potential integral {a.name} / {Path(p0['structureFile']).stem} = "
          f"{(v_new - np.median(v_new)).sum() / (v_old - np.median(v_old)).sum():.2f}")
    pdir, rdir = d / "pars_LSU_mouse", d / "results_LSU_mouse"
    pdir.mkdir(exist_ok=True)
    jobs = []
    for par in pars:
        new = {"structureFile": sp, "modelName": a.name, "outputDir": rdir / par.stem}
        out = pdir / par.name
        out.write_text("\n".join(prep.set_keys(par.read_text().splitlines(), new)) + "\n")
        prep.check_par(out)
        pn, _ = sio.read_params_file(out)
        po, _ = sio.read_params_file(par)
        diff = [k for k in pn if k not in new and str(pn[k]) != str(po[k])]
        if diff:
            raise SystemExit(f"{out}: unexpected differences {diff}")
        jobs.append(str(out))
    (d / "jobs_LSU_mouse.txt").write_text("\n".join(jobs) + "\n")
    print(f"{len(jobs)} searches (same tiles and settings as the 6EK0 control, template only) -> "
          f"{d / 'jobs_LSU_mouse.txt'} (~1.6 GPU hours each)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
