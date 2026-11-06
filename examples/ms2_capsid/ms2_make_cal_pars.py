#!/usr/bin/env python
"""MS2 searches again with the calibrated pixel size (from ms2_ctf_scan.py --fine).

For each micrograph of the first run:
  pixel     the calibrated value (default: median of diag/ms2_pixel_calibration.csv)
  defocus   cryoSPARC values x (pixel / 1.16)^2 (exact for a fit made at the wrong pixel, Cs 0.02 mm),
            centred between that and the defocus the virions themselves prefer (the scan), with
            planes every 50 nm over both values +-100 nm (at least +-200 nm, at most 11 planes;
            refinement then scans +-50 nm around each particle)
  angles    2.5 deg (icosahedral reduction keeps this cheap)
  template  the 2MS2 potential computed once at the new pixel size
Paths are written in full (no --path-map needed), so the jobs run with run_queue.sh or run_jobs.sh.

  cd $PYSMAP
  python examples/ms2_capsid/ms2_make_cal_pars.py --jobs $MS2_WORK/pars_a3.8/jobs10.txt --calib $MS2_WORK/diag/ms2_pixel_calibration.csv --model $MS2_DATA/models/2MS2-assembly1.cif --work $MS2_WORK
Writes work/models_cal/2MS2_capsid_SP.mrc, work/pars_cal/*.par, work/pars_cal/jobs.txt (-> work/results_cal/).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path

import numpy as np


def get(text: str, key: str):
    m = re.search(rf"^{key}\s+(.+)$", text, re.M)
    return m.group(1).strip() if m else None


def set_keys(text: str, values: dict) -> str:
    lines, done = [], set()
    for ln in text.splitlines():
        t = ln.split()
        if t and t[0] in values:
            if t[0] not in done:
                lines.append(f"{t[0]} {values[t[0]]}")
                done.add(t[0])
            continue
        lines.append(ln)
    lines += [f"{k} {v}" for k, v in values.items() if k not in done]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (smap)
    from smap import io as sio
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", required=True, help="jobs list of the first run")
    ap.add_argument("--calib", required=True, help="ms2_pixel_calibration.csv")
    ap.add_argument("--model", required=True, help="2MS2 assembly .cif")
    ap.add_argument("--work", required=True)
    ap.add_argument("--pixel", type=float, default=None, help="default: median of the calibration")
    ap.add_argument("--angle", type=float, default=2.5)
    ap.add_argument("--step-nm", type=float, default=50.0,
                    help="plane spacing; refinement scans +-50 nm around each particle's plane")
    ap.add_argument("--max-planes", type=int, default=11, help="GPU memory grows with planes (17 failed on 24 GB)")
    ap.add_argument("--path-map", default="/ms2=" + os.environ.get("MS2_DATA", str(Path.home() / "ms2_empiar10075")),
                    help="OLD=NEW prefix for paths in the first-run .par files")
    a = ap.parse_args(argv)
    work = Path(a.work).expanduser().resolve()
    src_map, dst_map = a.path_map.split("=", 1)
    fix = lambda p: str(p).replace(src_map, dst_map, 1) if str(p).startswith(src_map) else str(p)  # noqa: E731

    cal = {r["micrograph"]: r for r in csv.DictReader(open(Path(a.calib).expanduser()))}
    pixel = a.pixel or float(np.median([float(r["pixel_A"]) for r in cal.values()]))
    pars = [Path(p.strip()) for p in Path(a.jobs).expanduser().read_text().split() if p.strip()]
    first = pars[0].read_text()
    old_pix = float(get(first, "aPerPix"))
    s2 = (pixel / old_pix) ** 2
    print(f"pixel {old_pix} -> {pixel:.4f} A (defocus x {s2:.4f}); angles {a.angle} deg; planes every {a.step_nm:g} nm, at most {a.max_planes}")

    # template potential once, at the new pixel size
    from smap.structure import calculate_sp
    mdir = work / "models_cal"
    mdir.mkdir(parents=True, exist_ok=True)
    name = "2MS2_capsid"
    sp = mdir / f"{name}_SP.mrc"
    if not sp.exists():
        p, _ = sio.read_params_file(pars[0])
        p.update(structureFile=str(Path(a.model).expanduser().resolve()), modelName=name, aPerPix=pixel,
                 outputDir=str(mdir))
        print("template potential (CPU, a few minutes) ...", flush=True)
        calculate_sp(p, mdir)
        ep = mdir / f"{name}_EP.mrc"
        if ep.exists():
            ep.unlink()
    print(f"template: {sp}")

    odir, rdir = work / "pars_cal", work / "results_cal"
    odir.mkdir(exist_ok=True)
    jobs, n_planes = [], []
    print(f"{'micrograph':40s} {'cryoSPARC*s2':>12s} {'scan':>6s} {'centre':>7s} {'range':>9s} {'planes':>6s}")
    for par in pars:
        txt = par.read_text()
        stem = Path(get(txt, "outputDir")).name
        r = cal.get(stem)
        if r is None:
            print(f"{stem[:40]:40s} not in the calibration file: skipped")
            continue
        df1, df2, ang = (float(x) for x in get(txt, "defocus").split()[:3])
        d1, d2 = df1 * s2, df2 * s2
        cryo = (d1 + d2) / 2
        scan = float(r["defocus_A"]) * (pixel / float(r["pixel_A"])) ** 2
        centre = (cryo + scan) / 2
        half_nm = min(max(200.0, abs(scan - cryo) / 20 + 100.0), (a.max_planes - 1) / 2 * a.step_nm)
        T = math.ceil(2 * half_nm / a.step_nm) * a.step_nm
        shift = centre - cryo
        new = {"imageFile": fix(get(txt, "imageFile")), "structureFile": sp, "modelName": name,
               "symmetryFile": fix(get(txt, "symmetryFile")), "aPerPix": f"{pixel:.4f}",
               "defocus": f"{d1 + shift:.1f} {d2 + shift:.1f} {ang:.2f}", "angle_inc": a.angle,
               "T_sample": f"{T:g}", "df_inc": f"{a.step_nm:g}", "outputDir": rdir / stem}
        out = odir / f"{stem}.par"
        out.write_text(set_keys(txt, new))
        p, _ = sio.read_params_file(out)                       # read back with the port's reader
        bad = [k for k in ("imageFile", "structureFile", "symmetryFile", "outputDir") if str(p[k]) != str(new[k])]
        left = [k for k in ("imageFile", "structureFile", "symmetryFile", "outputDir") if str(p[k]).startswith(src_map)]
        if bad or left or abs(float(p["aPerPix"]) - pixel) > 1e-3:
            raise SystemExit(f"{out}: read back differs for {bad or left or 'aPerPix'}")
        for k in ("imageFile", "symmetryFile"):
            if not Path(new[k]).exists():
                raise SystemExit(f"{out}: {k} not found: {new[k]}")
        n = int(round(T / a.step_nm)) + 1
        n_planes.append(n)
        jobs.append(str(out))
        print(f"{stem[:40]:40s} {cryo / 1e4:12.2f} {scan / 1e4:6.2f} {centre / 1e4:7.2f} "
              f"{(centre - T * 5) / 1e4:4.2f}-{(centre + T * 5) / 1e4:4.2f} {n:6d}")
    (odir / "jobs.txt").write_text("\n".join(jobs) + "\n")
    n_rot = 288990 * (3.8 / a.angle) ** 3 / 60
    print(f"{len(jobs)} searches -> {odir / 'jobs.txt'}; about {n_rot:,.0f} orientations x {sum(n_planes)} planes in "
          f"total (TRiC tile: 288,990 x 5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
