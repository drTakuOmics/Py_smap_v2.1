#!/usr/bin/env python
"""Decoy-protein control: a ~1 MDa protein that cannot be in a mouse cell, searched exactly like TRiC.

Default decoy: lumazine synthase from Aquifex aeolicus (PDB 1HQK, biological assembly = 60-mer,
~1.0 MDa, hollow sphere ~15 nm; TRiC ~0.95 MDa, ~15 x 16 nm). Mammals make no riboflavin, so no
lumazine synthase or homologue exists in the sample. Same tiles, same settings, same 288,990
orientations as TRiC (no symmetry reduction, so the trial count and cutoff are identical).
Hits of the decoy at the TRiC cutoff estimate how many TRiC hits are false (target-decoy).

Why not apoferritin: ferritin is made by mouse cells (abundant in myeloid cells), so its hits could
be real; and it is ~0.5 MDa, half of TRiC. Why not a mirror: a projection does not see the hand.

  wget -O $WORK/models/1HQK-assembly1.cif.gz https://files.rcsb.org/download/1HQK-assembly1.cif.gz
  gunzip -f $WORK/models/1HQK-assembly1.cif.gz
  cd $PYSMAP
  python examples/decolace_tric/make_decoy_jobs.py --dest $WORK --model $WORK/models/1HQK-assembly1.cif
Writes models/LumSyn_1HQK_SP.mrc, pars_TRiC_decoy/, jobs_TRiC_decoy.txt (results -> results_TRiC_decoy/).
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np


def model_summary(path: Path) -> str:
    """Title, chains and atoms as pysmap reads them (ATOM records only: no water, no ligands)."""
    from smap.structure import read_cif_file
    txt = path.read_text(errors="replace")
    m = re.search(r"^_struct\.title\s+(.+)$", txt, re.M)
    title = m.group(1).strip().strip("'\"") if m else "?"
    at = read_cif_file(path)
    ext = np.ptp(at.xyz, axis=1) if at.xyz.size else np.zeros(3)
    return (f"{title}: {len(set(at.chain_ids))} chains, {at.xyz.shape[1]:,} atoms, "
            f"extent {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} A")


def main(argv=None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (smap)
    from smap import io as sio
    kit = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("prep", kit / "decolace_prep.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", required=True)
    ap.add_argument("--model", required=True, help="decoy assembly (.cif)")
    ap.add_argument("--name", default="LumSyn_1HQK")
    ap.add_argument("--label", default="lumazine_synthase_1HQK")
    ap.add_argument("--tiles", nargs="+", default=["00453", "00463", "00334"],
                    help="tile numbers (default: 24 clean TRiC hits; 453 and 463 also have phase-random controls)")
    a = ap.parse_args(argv)
    d = Path(a.dest).expanduser().resolve()
    model = Path(a.model).expanduser().resolve()
    print("decoy model:", model_summary(model))
    done = sorted(p.parent.name for p in (d / "results_TRiC").glob("*/.done"))
    stems = [s for s in done if any(f"_{k}_" in s for k in a.tiles)]
    if not stems:
        raise SystemExit(f"none of {a.tiles} finished in results_TRiC")
    src0 = d / "pars_TRiC" / f"{stems[0]}.par"
    p0, _ = sio.read_params_file(src0)
    pixel = float(p0["aPerPix"])
    sp = prep.make_sp(kit / "tric_decolace.par", model, a.name, pixel, d / "models")
    v_dec = sio.mr(sp).astype(np.float64)
    v_tric = sio.mr(p0["structureFile"]).astype(np.float64)
    print(f"potential integral decoy / TRiC = {v_dec.sum() / v_tric.sum():.2f} (mass ratio; aim ~1)")

    pdir = d / "pars_TRiC_decoy"
    pdir.mkdir(exist_ok=True)
    jobs = []
    for s in stems:
        src = d / "pars_TRiC" / f"{s}.par"
        new = {"structureFile": sp, "modelName": a.name, "decoy_label": a.label,
               "outputDir": d / "results_TRiC_decoy" / s}
        par = pdir / f"{s}.par"
        par.write_text("\n".join(prep.set_keys(src.read_text().splitlines(), new)) + "\n")
        prep.check_par(par)
        p_new, _ = sio.read_params_file(par)
        p_old, _ = sio.read_params_file(src)
        diff = [k for k in p_new if k not in new and str(p_new[k]) != str(p_old[k])]
        if diff or str(p_new["decoy_label"]) != a.label:
            raise SystemExit(f"{par}: settings differ from the TRiC search: {diff}")
        jobs.append(str(par))
    (d / "jobs_TRiC_decoy.txt").write_text("\n".join(jobs) + "\n")
    print(f"{len(jobs)} decoy searches, settings identical to TRiC except the template "
          f"-> {d / 'jobs_TRiC_decoy.txt'} (about {1.6 * len(jobs):.0f} GPU hours)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
