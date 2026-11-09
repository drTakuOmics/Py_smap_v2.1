#!/usr/bin/env python
"""Per-micrograph search .par files from a RELION particle STAR file (CTF per micrograph, and the
particles as a positive control).

Reads the particles (RELION 3.0 single table, or 3.1+ with data_optics / data_particles), groups
them by _rlnMicrographName and takes, per micrograph, the median DefocusU and DefocusV and the
circular mean of DefocusAngle (CTFFind convention, the same as RELION). Voltage and Cs come from
the STAR file; the pixel size is compared with the template. Micrograph file names can be mapped
to the files actually used (for example dose-weighted sums) with --rename.

    python tools/star_to_pars.py particles.star --template search.par --micrograph-dir /path/to/micrographs \\
        --out-dir pars --results-dir results --top 2 --rename '_st_Stk_corrected\\.mrc$=-a-DW_magcorrect.mrc'

Outputs in --out-dir: ``<name>.par`` and ``jobs.txt`` for the selected micrographs,
``ctf_table.csv`` (every micrograph: particles, defocus, spread of the particle defoci),
``coords/<name>.csv`` (particle centre x, y in micrograph pixels = coordinate minus the refined
origin shift; Euler angles rot, tilt, psi in degrees; defocus of each particle) and, with --url-prefix, ``download_urls.txt``.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # repo root (smap)
sys.path.insert(0, str(Path(__file__).resolve().parent))         # tools/
from cryosparc_ctf_to_pars import check_par, par_value, read_par, set_keys  # noqa: E402


def read_star(path) -> dict:
    """{block name: list of row dicts} for every loop_ in the STAR file."""
    blocks, name, keys, rows, in_loop = {}, None, [], None, False
    for raw in Path(path).read_text(errors="replace").splitlines():
        s = raw.strip()
        if s.startswith("data_"):
            name, in_loop, keys = s[5:] or "data", False, []
            continue
        if s == "loop_":
            in_loop, keys, rows = True, [], []
            blocks[name] = rows
            continue
        if not in_loop or not s or s.startswith("#"):
            continue
        if s.startswith("_"):
            keys.append(s.split()[0][1:])
            continue
        f = s.split()
        if len(f) == len(keys):
            rows.append(dict(zip(keys, f)))
    return blocks


def particles(path) -> list:
    """Particle rows with optics-group values merged in (RELION 3.1+); 3.0 rows unchanged."""
    b = read_star(path)
    rows = b.get("particles") or next((r for r in b.values() if r and "rlnMicrographName" in r[0]), [])
    if not rows:
        raise SystemExit(f"{path}: no particle table with rlnMicrographName")
    optics = {r.get("rlnOpticsGroup"): r for r in b.get("optics", [])}
    if optics:
        rows = [{**optics.get(r.get("rlnOpticsGroup"), {}), **r} for r in rows]
    return rows


def pixel_size(r: dict):
    for k in ("rlnMicrographPixelSize", "rlnMicrographOriginalPixelSize", "rlnImagePixelSize"):
        if k in r:
            return float(r[k])
    if "rlnDetectorPixelSize" in r and "rlnMagnification" in r:
        return float(r["rlnDetectorPixelSize"]) * 1e4 / float(r["rlnMagnification"])
    return None


def origin_px(r: dict) -> tuple:
    """Refined origin shift in micrograph pixels (RELION 3.0: pixels; 3.1+: Angstrom)."""
    if "rlnOriginX" in r:
        return float(r["rlnOriginX"]), float(r.get("rlnOriginY", 0))
    if "rlnOriginXAngst" in r:
        px = pixel_size(r) or 1.0
        return float(r["rlnOriginXAngst"]) / px, float(r.get("rlnOriginYAngst", 0)) / px
    return 0.0, 0.0


def renamed(name: str, rules) -> str:
    base = Path(name).name
    for pat, rep in rules:
        base = re.sub(pat, rep, base)
    return base


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("star", help="RELION particle STAR file")
    ap.add_argument("--template", required=True, help="search .par with the common settings")
    ap.add_argument("--micrograph-dir", required=True, help="folder with (or for) the micrographs to search")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--results-dir", required=True, help="outputDir of each search = <this>/<name>")
    ap.add_argument("--rename", action="append", default=[], metavar="REGEX=REPLACEMENT",
                    help="map STAR micrograph file names to the files searched (applied to the base name)")
    ap.add_argument("--top", type=int, default=None, help="only the N micrographs with the most particles")
    ap.add_argument("--names", nargs="+", default=None, help="only these micrographs (searched file stems)")
    ap.add_argument("--url-prefix", default=None, help="write download_urls.txt = prefix + file name")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a template key")
    a = ap.parse_args(argv)

    rules = [tuple(r.split("=", 1)) for r in a.rename]
    rows = particles(a.star)
    groups: dict = {}
    for r in rows:
        groups.setdefault(r["rlnMicrographName"], []).append(r)
    tmpl = read_par(a.template)
    apix = float((par_value(tmpl, "aPerPix") or ["nan"])[0])
    mdir = Path(a.micrograph_dir).expanduser().resolve()
    out = Path(a.out_dir).expanduser()
    (out / "coords").mkdir(parents=True, exist_ok=True)
    notes, table = [], []
    for mic, ps in groups.items():
        fname = renamed(mic, rules)
        u = np.array([float(p["rlnDefocusU"]) for p in ps])
        v = np.array([float(p["rlnDefocusV"]) for p in ps])
        ang = np.radians([float(p["rlnDefocusAngle"]) for p in ps])
        mean_ang = math.degrees(0.5 * math.atan2(np.sin(2 * ang).mean(), np.cos(2 * ang).mean())) % 180
        dfm = (u + v) / 2
        table.append(dict(micrograph=Path(fname).stem, star_name=mic, particles=len(ps),
                          df1_A=round(float(np.median(u)), 1), df2_A=round(float(np.median(v)), 1),
                          angle_deg=round(mean_ang, 2), particle_df_range_A=round(float(np.ptp(dfm)), 1),
                          particle_df_sd_A=round(float(dfm.std()), 1), voltage_kV=ps[0].get("rlnVoltage"),
                          cs_mm=ps[0].get("rlnSphericalAberration"), pixel_A=pixel_size(ps[0]), file=fname))
        with open(out / "coords" / f"{Path(fname).stem}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["x_px", "y_px", "rot", "tilt", "psi", "dfU_A", "dfV_A"])
            for p in ps:
                ox, oy = origin_px(p)
                w.writerow([f"{float(p['rlnCoordinateX']) - ox:.2f}", f"{float(p['rlnCoordinateY']) - oy:.2f}",
                            p.get("rlnAngleRot"), p.get("rlnAngleTilt"), p.get("rlnAnglePsi"),
                            p.get("rlnDefocusU"), p.get("rlnDefocusV")])
    table.sort(key=lambda t: (-t["particles"], t["micrograph"]))
    sel = [t for t in table if not a.names or t["micrograph"] in a.names]
    if a.top:
        sel = sel[: a.top]

    px = {t["pixel_A"] for t in table if t["pixel_A"]}
    if px and any(abs(p - apix) > 1e-3 for p in px):
        notes.append(f"pixel size in the STAR {sorted(px)} differs from template aPerPix {apix}")
    jobs = []
    for t in sel:
        name = t["micrograph"]
        img, res = mdir / t["file"], Path(a.results_dir).expanduser().resolve() / name
        vals = {"imageFile": img, "defocus": f"{t['df1_A']:.1f} {t['df2_A']:.1f} {t['angle_deg']:.2f}",
                "defocus_format": "CTFFind", "outputDir": res}
        if t["voltage_kV"]:
            vals["V_acc"] = f"{float(t['voltage_kV']) * 1000:.1f}"
        if t["cs_mm"]:
            vals["Cs"] = f"{float(t['cs_mm']) * 1e-3:g}"
        for kv in a.set:
            k, v = kv.split("=", 1)
            vals[k] = v
        par = out / f"{name}.par"
        par.write_text("\n".join(set_keys(tmpl, vals)) + "\n")
        check_par(par, img, res, (t["df1_A"], t["df2_A"], t["angle_deg"]))
        jobs.append(str(par.resolve()))
        if not img.exists():
            notes.append(f"{img.name}: not downloaded yet")
    with open(out / "ctf_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    (out / "jobs.txt").write_text("\n".join(jobs) + ("\n" if jobs else ""))
    if a.url_prefix:
        (out / "download_urls.txt").write_text("".join(f"{a.url_prefix.rstrip('/')}/{t['file']}\n" for t in sel))

    n = np.array([t["particles"] for t in table])
    spread = np.array([t["particle_df_range_A"] for t in table])
    df = np.array([(t["df1_A"] + t["df2_A"]) / 2 for t in table])
    print(f"{len(rows)} particles on {len(table)} micrographs (median {np.median(n):.0f} per micrograph); "
          f"{len(jobs)} .par written to {out}")
    print(f"mean defocus {df.min():.0f}-{df.max():.0f} A (median {np.median(df):.0f}); particle defocus range per "
          f"micrograph median {np.median(spread):.0f} A (sample thickness), 90th percentile {np.percentile(spread, 90):.0f} A")
    print(f"voltage {table[0]['voltage_kV']} kV, Cs {table[0]['cs_mm']} mm, pixel {sorted(px)} A; template aPerPix {apix}")
    for t in sel:
        print(f"  {t['micrograph']}: {t['particles']} particles, defocus {t['df1_A']:.0f} / {t['df2_A']:.0f} A, "
              f"angle {t['angle_deg']:.1f}, particle defocus range {t['particle_df_range_A']:.0f} A")
    for s in dict.fromkeys(notes):
        print("CHECK", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
