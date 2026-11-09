#!/usr/bin/env python
"""Detections of a search against a reference particle list: found, missed and extra at several cutoffs.

    python tools/match_positions.py RESULTS/<micrograph>/ REFERENCE.csv [--radius-A 40] [--cutoffs 7 7.5 8]

REFERENCE.csv has x_px, y_px in original micrograph pixels (x = column, y = row), for example
coords/<name>.csv from tools/star_to_pars.py. The search's particles.csv gives x_orig0_est,
y_orig0_est in the same frame. Detections and references are paired one to one within --radius-A.
References closer to the image edge than the search margin (margin_pix) cannot be reported by the
search and are counted separately. The pairing is also tried with y flipped; a warning is printed
if that pairs more (a coordinate-convention problem rather than a detection result).
Writes <folder>/match_<reference>.csv: every detection with its score, paired reference and distance.
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", help="search output folder (particles.csv, search_status.json)")
    ap.add_argument("reference", help="CSV with x_px, y_px (original micrograph pixels)")
    ap.add_argument("--radius-A", type=float, default=40.0, help="pairing distance")
    ap.add_argument("--cutoffs", type=float, nargs="*", default=[7.0, 7.5, 8.0, 9.0, 10.0])
    a = ap.parse_args(argv)
    f = Path(a.folder)
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
    ref = np.array([[float(r["x_px"]), float(r["y_px"])] for r in csv.DictReader(open(a.reference))]).reshape(-1, 2)
    r_px = a.radius_A / a_orig
    inside = np.ones(len(ref), bool)
    if shape:
        ny, nx = shape
        inside = ((ref[:, 0] >= margin) & (ref[:, 0] < nx - margin) & (ref[:, 1] >= margin) & (ref[:, 1] < ny - margin))
    cuts = sorted(set(([round(cut0, 2)] if cut0 else []) + list(a.cutoffs)))
    print(f"{f.name}: {len(rows)} refined candidates; {len(ref)} reference particles, {int(inside.sum())} inside the "
          f"search margin ({margin:.0f} px); pairing within {a.radius_A:g} A")
    print(f"{'cutoff':>7s} {'detected':>9s} {'paired':>7s} {'extra':>6s} {'missed':>7s} {'recall':>7s}")
    for c in cuts:
        sel = np.nonzero(score >= c)[0]
        pr = pair(det[sel], ref, r_px)
        hit = np.zeros(len(ref), bool)
        hit[[j for _, j in pr]] = True
        tag = "  (search cutoff)" if cut0 and abs(c - cut0) < 0.006 else ""
        print(f"{c:7.2f} {len(sel):9d} {len(pr):7d} {len(sel) - len(pr):6d} {int((inside & ~hit).sum()):7d} "
              f"{100 * (inside & hit).sum() / max(inside.sum(), 1):6.0f}%{tag}")
    c = cut0 or min(cuts)
    sel = np.nonzero(score >= c)[0]
    pr = pair(det[sel], ref, r_px)
    if pr:
        dxy = np.array([det[sel][i] - ref[j] for i, j in pr]) * a_orig
        print(f"offset detection - reference at {c:.2f}: dx {dxy[:, 0].mean():+.1f} A, dy {dxy[:, 1].mean():+.1f} A "
              f"(sd {dxy[:, 0].std():.1f}, {dxy[:, 1].std():.1f} A)")
    if shape:
        flip = ref.copy()
        flip[:, 1] = shape[0] - 1 - flip[:, 1]
        n_flip = len(pair(det[sel], flip, r_px))
        if n_flip > len(pr):
            print(f"WARNING: with y flipped {n_flip} pairs instead of {len(pr)}: the reference uses another y convention")
    pr_all = dict(pair(det, ref, r_px))
    out = f / f"match_{Path(a.reference).stem}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["num_id", "score_final", "x_px", "y_px", "reference_index", "distance_A"])
        for i, r in enumerate(rows):
            j = pr_all.get(i, -1)
            dist = np.hypot(*(det[i] - ref[j])) * a_orig if j >= 0 else ""
            w.writerow([r["num_id"], r["score_final"], f"{det[i, 0]:.1f}", f"{det[i, 1]:.1f}", j,
                        f"{dist:.1f}" if j >= 0 else ""])
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
