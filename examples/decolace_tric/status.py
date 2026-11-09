#!/usr/bin/env python
"""One table of every search in the DeCo-LACE TRiC work: what it is, where it is, how far it got.

  python examples/decolace_tric/status.py    (working folder: $WORK, default ~/decolace)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

H = Path.home()
D = Path(os.environ.get("WORK", H / "decolace")).expanduser()          # DeCo-LACE working folder

ROWS = [
    # name, template, image, jobs file, results folder, what it tests
    ("TRiC closed", "TRiC closed (7NVL, mouse)", "as deposited", D / "jobs_TRiC.txt", D / "results_TRiC",
     "main search"),
    ("LSU control", "LSU (6EK0)", "as deposited", D / "jobs_LSU_control.txt", D / "results_LSU_control",
     "pipeline check against the deposited ribosomes"),
    ("LSU mouse (6SWA)", "LSU 6SWA, no Ebp1", "as deposited", D / "jobs_LSU_mouse.txt", D / "results_LSU_mouse",
     "mouse template, same tiles"),
    ("LSU tile 520 filled", "LSU (6EK0)", "platinum region -> noise", None, D / "results_controls" / "LSU_00520_filled",
     "artifact zone filled with noise"),
    ("TRiC tile 520 filled", "TRiC closed", "platinum region -> noise", None, D / "results_controls" / "TRiC_00520_filled",
     "artifact zone filled with noise"),
    ("TRiC noise 463", "TRiC closed", "phase-randomised", None, D / "results_controls" / "TRiC_00463_phaserand1",
     "noise rate after refinement"),
    ("TRiC noise 453", "TRiC closed", "phase-randomised", None, D / "results_controls" / "TRiC_00453_phaserand1",
     "noise rate after refinement"),
    ("TRiC decoy", "lumazine synthase (1HQK)", "as deposited", D / "jobs_TRiC_decoy.txt", D / "results_TRiC_decoy",
     "decoy protein, same tiles"),
    ("TRiC open", "TRiC open (6NR8 hybrid)", "as deposited", D / "jobs_TRiC_open.txt", D / "results_TRiC_open",
     "closed vs open state on the same tiles"),
]


def progress(log: Path) -> str:
    if not log.exists():
        return ""
    tail = log.read_bytes()[-4000:].decode("utf-8", "replace").splitlines()
    for ln in reversed(tail):
        m = re.search(r"(\d+)/(\d+) orientations", ln)
        if m:
            return f"{100 * int(m.group(1)) / int(m.group(2)):.0f}% searched"
        if "particle" in ln:
            return "refining"
    return "started"


def state(folder: Path, jobs: Path | None) -> tuple:
    if jobs is not None:
        if not jobs.exists():
            return "not prepared", ""
        pars = [Path(p.strip()) for p in jobs.read_text().split() if p.strip()]
        outs = []
        for p in pars:
            if p.exists():
                m = re.search(r"^outputDir\s+(\S+)", p.read_text(), re.M)
                outs.append(Path(m.group(1)) if m else None)
        done = sum(1 for o in outs if o and ((o / ".done").exists() or (o / "particles.csv").exists()))
        running = [o for o in outs if o and o.exists() and not (o / ".done").exists() and not (o / "particles.csv").exists()]
        extra = progress(Path(str(running[0]) + ".log")) if running else ""
        return f"{done}/{len(pars)} done", extra
    if (folder / ".done").exists() or (folder / "particles.csv").exists():
        return "done", ""
    if folder.exists():
        return "running", progress(Path(str(folder) + ".log"))
    return "queued", ""


def main() -> int:
    print(f"{'search':22s} {'template':26s} {'image':26s} {'state':14s} {'':16s} folder / question")
    for name, tmpl, img, jobs, folder, why in ROWS:
        st, extra = state(folder, jobs)
        print(f"{name:22s} {tmpl:26s} {img:26s} {st:14s} {extra:16s} {str(folder).replace(str(H), '~')}  ({why})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
