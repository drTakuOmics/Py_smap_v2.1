#!/usr/bin/env python
"""Open-state TRiC search on the tiles already searched with the closed template.

Template: ``open_hybrid_mouse.cif`` = the mouse closed model (7NVL + mouse sequences) cut into
equatorial / intermediate / apical domains per subunit and each domain superposed on the
open-state scaffold 6NR8 (human TRiC-prefoldin, open, 7.8 Å; prefoldin not included), hinges
relaxed. Same atoms and side chains as the closed template, open domain positions.

  cd $PYSMAP
  python examples/decolace_tric/make_open_jobs.py --dest $WORK --model examples/decolace_tric/open_hybrid_mouse.cif
Writes models/TRiC_open_mouse_SP.mrc, pars_TRiC_open/, jobs_TRiC_open.txt (results -> results_TRiC_open/).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (smap)
    kit = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("prep", kit / "decolace_prep.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", required=True)
    ap.add_argument("--model", default=str(Path(__file__).resolve().parent / "models" / "open_hybrid_mouse.cif.gz"))
    ap.add_argument("--name", default="TRiC_open_mouse")
    ap.add_argument("--exclude", nargs="*", default=["00520"], help="tile numbers to skip (image artifacts)")
    a = ap.parse_args(argv)
    d = Path(a.dest).expanduser().resolve()
    tiles = {t["stem"]: t for t in csv.DictReader(open(d / "tiles.csv"))}
    done = sorted(p.parent.name for p in (d / "results_TRiC").glob("*/.done"))
    done = [s for s in done if not any(f"_{k}_" in s for k in a.exclude)]
    if not done:
        raise SystemExit("no finished closed-TRiC tiles in results_TRiC")
    pixel = float(np.median([float(tiles[s]["pixel"]) for s in done if s in tiles]))
    print(f"template potential {a.name} at {pixel:.4f} A ...", flush=True)
    sp = prep.make_sp(kit / "tric_decolace.par", Path(a.model).expanduser().resolve(), a.name, pixel, d / "models")
    pdir = d / "pars_TRiC_open"
    pdir.mkdir(exist_ok=True)
    jobs = []
    for s in done:
        src = d / "pars_TRiC" / f"{s}.par"
        lines = prep.set_keys(src.read_text().splitlines(), {
            "structureFile": sp, "modelName": a.name, "outputDir": d / "results_TRiC_open" / s})
        par = pdir / f"{s}.par"
        par.write_text("\n".join(lines) + "\n")
        prep.check_par(par)
        jobs.append(str(par))
    (d / "jobs_TRiC_open.txt").write_text("\n".join(jobs) + "\n")
    skipped = [k for k in a.exclude if any(f"_{k}_" in p.parent.name for p in (d / "results_TRiC").glob("*/.done"))]
    print(f"{len(jobs)} open-state searches (same tiles, same settings; skipped {skipped}) -> {d / 'jobs_TRiC_open.txt'}")
    print(f"about {1.6 * len(jobs):.0f} GPU hours at the closed-template speed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
