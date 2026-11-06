#!/usr/bin/env python
"""Per-micrograph search .par files from a cryoSPARC CTF job (Patch CTF or CTFFIND4).

Reads the exposure .cs files of the job (numpy structured arrays; the CTF file and the
passthrough file are joined on ``uid``), takes ``ctf/df1_A``, ``ctf/df2_A``, ``ctf/df_angle_rad``
and writes, for every micrograph, a copy of a template .par with ``imageFile``, ``defocus``
(CTFFind format: df1 A, df2 A, angle deg) and ``outputDir`` set. Micrographs are matched to
the original files by name (cryoSPARC's numeric import prefix is removed).

    python tools/cryosparc_ctf_to_pars.py J5/J5_exposures_ctf_estimated.cs J5/J5_passthrough_exposures.cs \\
        --template ms2_capsid.par --micrograph-dir ~/data/ms2/micrographs \\
        --out-dir pars --results-dir results [--max-fit-A 8]

Outputs in --out-dir: ``<name>.par``, ``ctf_table.csv`` (all micrographs, with the reason for any
exclusion) and ``jobs.txt`` (the .par files to run). Checks pixel size, voltage and Cs against the
template .par and prints the differences.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # repo root (smap) when run as a script

import argparse
import csv
import math
import re

import numpy as np

PREFIX = re.compile(r"^\d{6,}_")          # cryoSPARC import prefix (uid_)


def load_cs(paths) -> dict:
    """Join the fields of several .cs files on uid -> {field: array} (rows ordered by uid)."""
    cols: dict = {}
    uid = None
    for p in paths:
        a = np.load(p, allow_pickle=False)
        order = np.argsort(a["uid"])
        a = a[order]
        if uid is None:
            uid = a["uid"]
        elif not np.array_equal(uid, a["uid"]):
            common = np.intersect1d(uid, a["uid"])
            keep_old = np.isin(uid, common)
            cols = {k: v[keep_old] for k, v in cols.items()}
            uid = uid[keep_old]
            a = a[np.isin(a["uid"], common)]
        for k in a.dtype.names:
            cols.setdefault(k, a[k])
    return cols


def _s(x) -> str:
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)


def original_name(cs_path: str) -> str:
    return PREFIX.sub("", Path(cs_path).name)


def read_par(path) -> list:
    return Path(path).read_text().splitlines()


def par_value(lines, key):
    for ln in lines:
        t = ln.split()
        if t and t[0] == key and len(t) > 1:
            return t[1:]
    return None


def set_keys(lines, values: dict) -> list:
    out, done = [], set()
    for ln in lines:
        t = ln.split()
        if t and t[0] in values:
            if t[0] not in done:
                out.append(f"{t[0]} {values[t[0]]}")
                done.add(t[0])
            continue
        out.append(ln)
    for k, v in values.items():
        if k not in done:
            out.append(f"{k} {v}")
    return out


def check_par(par, img, res, df):
    """Read the written .par back with the port's reader (v2.1 rules: a key matches anywhere in
    a line, lines with '#' are skipped) and confirm the three per-micrograph values."""
    from smap import io as sio
    p, _ = sio.read_params_file(par)
    got = np.atleast_1d(np.asarray(p["defocus"], dtype=np.float64))
    ok = (str(p["imageFile"]) == str(img) and str(p["outputDir"]) == str(res)
          and got.size == 3 and np.allclose(got, np.round(df, 2), atol=0.06)
          and p.get("defocus_format") == "CTFFind")
    if not ok:
        raise SystemExit(f"{par}: values do not read back (imageFile {p['imageFile']}, outputDir "
                         f"{p['outputDir']}, defocus {got}); check the template and paths")
    # a key name inside a path on another line would set that key (v2.1 reader): catch it
    lines = [ln.split() for ln in Path(par).read_text().splitlines() if ln.strip() and "#" not in ln]
    given = {t[0]: " ".join(t[1:]) for t in lines}
    for f in sio._SEARCH_GLOBAL_STRING_FIELDS:
        v, d = p.get(f), sio.SEARCH_GLOBAL_DEFAULTS.get(f)
        if v != d and str(v) != given.get(f):
            raise SystemExit(f"{par}: key '{f}' reads as '{v}' (its name occurs inside another line); "
                             "rename the folder or file that contains it")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cs", nargs="+", help="exposure .cs file(s) of the CTF job (CTF + passthrough)")
    ap.add_argument("--template", required=True, help="search .par with the common settings")
    ap.add_argument("--micrograph-dir", required=True, help="folder with the original micrographs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--results-dir", required=True, help="outputDir of each search = <this>/<name>")
    ap.add_argument("--max-fit-A", type=float, default=None,
                    help="exclude micrographs whose CTF fit resolution is worse than this")
    ap.add_argument("--max-astig-A", type=float, default=None, help="exclude |df1-df2| above this")
    ap.add_argument("--flip-angle", action="store_true",
                    help="use -angle (only if tools/ctf_check.py shows the flipped sign fits better)")
    ap.add_argument("--pixel", type=float, default=None,
                    help="aPerPix for the search (e.g. a calibrated value); defocus is scaled by "
                         "(pixel / CTF-fit pixel)^2 so the CTF in pixel units is unchanged")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a template key (repeatable), e.g. --set angle_inc=2.5")
    ap.add_argument("--names", nargs="+", default=None, help="only these micrographs (file stems)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N micrographs (by name)")
    a = ap.parse_args(argv)

    cs = load_cs(a.cs)
    need = ("ctf/df1_A", "ctf/df2_A", "ctf/df_angle_rad", "micrograph_blob/path")
    miss = [k for k in need if k not in cs]
    if miss:
        raise SystemExit(f"missing fields {miss}; pass both the CTF .cs and the passthrough .cs")
    tmpl = read_par(a.template)
    over = dict(kv.split("=", 1) for kv in a.set)
    if over:
        tmpl = set_keys(tmpl, over)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mdir = Path(a.micrograph_dir).expanduser()

    # optics checks against the template
    notes = []
    apix = float(par_value(tmpl, "aPerPix")[0])
    for field, key, conv, unit in (("micrograph_blob/psize_A", "aPerPix", 1.0, "A/px"),
                                   ("ctf/accel_kv", "V_acc", 1e3, "V"),
                                   ("ctf/cs_mm", "Cs", 1e-3, "m")):
        if field in cs and par_value(tmpl, key):
            v = np.unique(np.round(cs[field].astype(np.float64) * conv, 9))
            p = float(par_value(tmpl, key)[0])
            if v.size != 1 or not math.isclose(v[0], p, rel_tol=1e-3):
                notes.append(f"{key}: cryoSPARC {v.tolist()} {unit} vs template {p} {unit}")
    fit_pix = apix
    if "micrograph_blob/psize_A" in cs and np.unique(cs["micrograph_blob/psize_A"]).size == 1:
        fit_pix = float(cs["micrograph_blob/psize_A"][0])
    scale = 1.0
    if a.pixel is not None:
        scale = (a.pixel / fit_pix) ** 2
        tmpl = set_keys(tmpl, {"aPerPix": a.pixel})
        apix = a.pixel
        notes.append(f"aPerPix {a.pixel} (CTF fitted at {fit_pix}): defocus x {scale:.5f}")

    rows, jobs = [], []
    n = len(cs["uid"])
    names = [original_name(_s(x)) for x in cs["micrograph_blob/path"]]
    order = sorted(range(n), key=lambda j: names[j])
    if a.names:
        order = [j for j in order if Path(names[j]).stem in set(a.names)]
    if a.limit:
        order = order[:a.limit]
    for i in order:
        name = names[i]
        stem = Path(name).stem
        img = mdir / name
        df1, df2 = float(cs["ctf/df1_A"][i]) * scale, float(cs["ctf/df2_A"][i]) * scale
        ang = math.degrees(float(cs["ctf/df_angle_rad"][i])) * (-1 if a.flip_angle else 1)
        fit = float(cs["ctf_stats/ctf_fit_to_A"][i]) if "ctf_stats/ctf_fit_to_A" in cs else float("nan")
        why = ""
        if not img.exists():
            why = "micrograph not found"
        elif a.max_fit_A is not None and not fit <= a.max_fit_A:
            why = f"fit {fit:.1f} A > {a.max_fit_A}"
        elif a.max_astig_A is not None and abs(df1 - df2) > a.max_astig_A:
            why = f"astigmatism {abs(df1 - df2):.0f} A > {a.max_astig_A}"
        rows.append(dict(name=stem, image=str(img), df1_A=round(df1, 1), df2_A=round(df2, 1),
                         angle_deg=round(ang, 2), mean_df_A=round((df1 + df2) / 2, 1),
                         astig_A=round(abs(df1 - df2), 1), ctf_fit_A=round(fit, 2), excluded=why))
        if why:
            continue
        par = out / f"{stem}.par"
        res = Path(a.results_dir).expanduser().resolve() / stem
        lines = set_keys(tmpl, {"imageFile": img.resolve(),
                                "defocus": f"{df1:.1f} {df2:.1f} {ang:.2f}",
                                "outputDir": res})
        par.write_text("\n".join(lines) + "\n")
        check_par(par, img.resolve(), res, (df1, df2, ang))
        jobs.append(str(par.resolve()))

    with open(out / "ctf_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "jobs.txt").write_text("\n".join(jobs) + ("\n" if jobs else ""))

    df = np.array([r["mean_df_A"] for r in rows])
    ast = np.array([r["astig_A"] for r in rows])
    print(f"{n} micrographs in the .cs, {len(rows)} selected; {len(jobs)} .par written to {out} "
          f"({len(rows) - len(jobs)} excluded, see ctf_table.csv)")
    print(f"mean defocus {df.min():.0f}-{df.max():.0f} A (median {np.median(df):.0f}); "
          f"astigmatism median {np.median(ast):.0f} A, max {ast.max():.0f} A; template aPerPix {apix}")
    for s in notes:
        print("CHECK", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
