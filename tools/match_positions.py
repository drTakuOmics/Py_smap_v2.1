#!/usr/bin/env python
"""Detections of a search against a reference particle list: found, missed and extra at several cutoffs.

    python tools/match_positions.py RESULTS/<micrograph>/ REFERENCE.csv [--radius-A 40] [--cutoffs 7 7.5 8]
    python tools/match_positions.py RESULTS/*/ --reference-dir COORDS/ --summary detections.csv

REFERENCE.csv has x_px, y_px in original micrograph pixels (x = column, y = row), for example
coords/<name>.csv from tools/star_to_pars.py. The search's particles.csv gives x_orig0_est,
y_orig0_est in the same frame. Detections and references are paired one to one within --radius-A.
References closer to the image edge than the search margin (margin_pix) cannot be reported by the
search and are counted separately. The pairing is also tried with y flipped; a warning is printed
if that pairs more (a coordinate-convention problem rather than a detection result).
Writes <folder>/match_<reference>.csv: every detection with its score, paired reference and distance.
With several folders (--reference-dir: lists named <folder name>.csv, as written by star_to_pars.py),
prints one line per micrograph at its search cutoff and the totals.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def par_value(text: str, key: str):
    """Values of one key in the text of a .par file (None if the key is absent)."""
    m = re.search(rf"^{key}\s+(.+)$", text, re.M)
    return m.group(1).split() if m else None


def pair(det: np.ndarray, ref: np.ndarray, r_px: float) -> list:
    """One-to-one pairs (i_det, i_ref) closer than r_px, minimising the summed distance."""
    if not len(det) or not len(ref):
        return []
    d = np.hypot(det[:, None, 0] - ref[None, :, 0], det[:, None, 1] - ref[None, :, 1])
    cost = np.where(d <= r_px, d, 1e9)
    i, j = linear_sum_assignment(cost)
    return [(a, b) for a, b in zip(i, j) if d[a, b] <= r_px]


def match_one(f: Path, reference: Path, radius_A: float, cutoffs, verbose: bool = True) -> dict:
    """Pair one search with its reference list; print the per-cutoff table (verbose), write
    <folder>/match_<reference>.csv and return the numbers at the search cutoff."""
    st = json.loads((f / "search_status.json").read_text())
    txt = st.get("provenance", {}).get("params_text", "")
    a_orig = float(par_value(txt, "aPerPix")[0])
    a_srch = float((par_value(txt, "aPerPix_search") or [a_orig])[0])
    margin = float((par_value(txt, "margin_pix") or ["0"])[0]) * a_srch / a_orig
    cut0 = st.get("gaussian_threshold")
    shape = None
    img = (par_value(txt, "imageFile") or [""])[0]
    if img and Path(img).exists():
        import mrcfile
        with mrcfile.open(img, permissive=True, header_only=True) as m:
            shape = (int(m.header.ny), int(m.header.nx))
    rows = list(csv.DictReader(open(f / "particles.csv")))
    det = np.array([[float(r["x_orig0_est"]), float(r["y_orig0_est"])] for r in rows]).reshape(-1, 2)
    score = np.array([float(r["score_final"]) for r in rows])
    ref = np.array([[float(r["x_px"]), float(r["y_px"])] for r in csv.DictReader(open(reference))]).reshape(-1, 2)
    r_px = radius_A / a_orig
    inside = np.ones(len(ref), bool)
    if shape:
        ny, nx = shape
        inside = ((ref[:, 0] >= margin) & (ref[:, 0] < nx - margin) & (ref[:, 1] >= margin) & (ref[:, 1] < ny - margin))
    cuts = sorted(set(([round(cut0, 2)] if cut0 else []) + list(cutoffs)))
    if verbose:
        print(f"{f.name}: {len(rows)} refined candidates; {len(ref)} reference particles, {int(inside.sum())} inside "
              f"the search margin ({margin:.0f} px); pairing within {radius_A:g} A")
        print(f"{'cutoff':>7s} {'detected':>9s} {'paired':>7s} {'extra':>6s} {'missed':>7s} {'recall':>7s}")
    for c in cuts:
        sel = np.nonzero(score >= c)[0]
        pr = pair(det[sel], ref, r_px)
        hit = np.zeros(len(ref), bool)
        hit[[j for _, j in pr]] = True
        tag = "  (search cutoff)" if cut0 and abs(c - cut0) < 0.006 else ""
        if verbose:
            print(f"{c:7.2f} {len(sel):9d} {len(pr):7d} {len(sel) - len(pr):6d} {int((inside & ~hit).sum()):7d} "
                  f"{100 * (inside & hit).sum() / max(inside.sum(), 1):6.0f}%{tag}")
    c = cut0 or min(cuts)
    sel = np.nonzero(score >= c)[0]
    pr = pair(det[sel], ref, r_px)
    hit = np.zeros(len(ref), bool)
    hit[[j for _, j in pr]] = True
    paired_det = {sel[i] for i, _ in pr}
    extra = [k for k in sel if k not in paired_det]
    res = dict(micrograph=f.name, cutoff=round(float(c), 2), refined=len(rows), detected=len(sel),
               reference=len(ref), inside=int(inside.sum()), found=int((inside & hit).sum()),
               missed=int((inside & ~hit).sum()), extra=len(extra),
               extra_ge10=int(sum(score[k] >= 10 for k in extra)),
               paired_score_median=round(float(np.median([score[sel[i]] for i, _ in pr])), 2) if pr else "",
               extra_score_median=round(float(np.median(score[extra])), 2) if extra else "")
    if pr:
        dxy = np.array([det[sel][i] - ref[j] for i, j in pr]) * a_orig
        res.update(dx_A=round(float(dxy[:, 0].mean()), 2), dy_A=round(float(dxy[:, 1].mean()), 2),
                   sd_dx_A=round(float(dxy[:, 0].std()), 2), sd_dy_A=round(float(dxy[:, 1].std()), 2))
        if verbose:
            print(f"offset detection - reference at {c:.2f}: dx {dxy[:, 0].mean():+.1f} A, dy {dxy[:, 1].mean():+.1f} A "
                  f"(sd {dxy[:, 0].std():.1f}, {dxy[:, 1].std():.1f} A)")
    if shape:
        flip = ref.copy()
        flip[:, 1] = shape[0] - 1 - flip[:, 1]
        n_flip = len(pair(det[sel], flip, r_px))
        if n_flip > len(pr):
            print(f"WARNING {f.name}: with y flipped {n_flip} pairs instead of {len(pr)}: the reference uses another "
                  "y convention")
    pr_all = dict(pair(det, ref, r_px))
    out = f / f"match_{Path(reference).stem}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["num_id", "score_final", "x_px", "y_px", "reference_index", "distance_A"])
        for i, r in enumerate(rows):
            j = pr_all.get(i, -1)
            dist = np.hypot(*(det[i] - ref[j])) * a_orig if j >= 0 else ""
            w.writerow([r["num_id"], r["score_final"], f"{det[i, 0]:.1f}", f"{det[i, 1]:.1f}", j,
                        f"{dist:.1f}" if j >= 0 else ""])
    if verbose:
        print(f"-> {out}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="FOLDER REFERENCE.csv, or several folders with --reference-dir")
    ap.add_argument("--reference-dir", default=None, help="reference lists named <folder name>.csv (several folders)")
    ap.add_argument("--summary", default=None, help="CSV with one row per folder (several folders)")
    ap.add_argument("--radius-A", type=float, default=40.0, help="pairing distance")
    ap.add_argument("--cutoffs", type=float, nargs="*", default=[7.0, 7.5, 8.0, 9.0, 10.0])
    a = ap.parse_args(argv)
    if a.reference_dir is None:
        if len(a.paths) != 2:
            ap.error("give FOLDER REFERENCE.csv, or folders with --reference-dir")
        match_one(Path(a.paths[0]), Path(a.paths[1]), a.radius_A, a.cutoffs)
        return 0
    rows = []
    for f in map(Path, a.paths):
        ref = Path(a.reference_dir) / f"{f.name}.csv"
        if not (f / "particles.csv").exists() or not ref.exists():
            print(f"{f.name}: skipped ({'no particles.csv' if not (f / 'particles.csv').exists() else 'no reference'})")
            continue
        rows.append(match_one(f, ref, a.radius_A, a.cutoffs, verbose=False))
    if not rows:
        return 1
    print(f"{'micrograph':38s} {'refined':>7s} {'reference':>9s} {'found':>6s} {'missed':>6s} {'recall':>6s} "
          f"{'extra':>6s} {'>=10':>5s}")
    for r in rows:
        print(f"{r['micrograph'][:38]:38s} {r['refined']:7d} {r['inside']:9d} {r['found']:6d} {r['missed']:6d} "
              f"{100 * r['found'] / max(r['inside'], 1):5.0f}% {r['extra']:6d} {r['extra_ge10']:5d}")
    tot = {k: sum(r[k] for r in rows) for k in ("refined", "inside", "found", "missed", "extra", "extra_ge10")}
    print(f"{'total (' + str(len(rows)) + ' micrographs)':38s} {tot['refined']:7d} {tot['inside']:9d} {tot['found']:6d} "
          f"{tot['missed']:6d} {100 * tot['found'] / max(tot['inside'], 1):5.0f}% {tot['extra']:6d} {tot['extra_ge10']:5d}")
    sd = [r for r in rows if "sd_dx_A" in r]
    if sd:
        print(f"position difference to the reference: mean dx {np.mean([r['dx_A'] for r in sd]):+.2f} A, dy "
              f"{np.mean([r['dy_A'] for r in sd]):+.2f} A; per-micrograph SD median {np.median([r['sd_dx_A'] for r in sd]):.2f}"
              f" / {np.median([r['sd_dy_A'] for r in sd]):.2f} A")
    cuts = sorted({r["cutoff"] for r in rows})
    print(f"search cutoffs {cuts[0]:g}-{cuts[-1]:g}; 'reference' = reference particles inside the search margin")
    if a.summary:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(a.summary, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"-> {a.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
