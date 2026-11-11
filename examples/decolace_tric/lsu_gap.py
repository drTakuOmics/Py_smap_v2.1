#!/usr/bin/env python
"""Search z-score at every deposited ribosome (LSU) position, by detection class (CPU, seconds).

For each LSU search (results_LSU_control/*, results_controls/LSU_*_filled) every deposited peak
(cisTEM, SNR >= 7.75) is put in one class:
  found        a pysmap detection >= cutoff within 20 A
  refined      a candidate was refined there but its final score is below the cutoff
  gate 6-7     search z 6-7: never refined (refinement starts at optThr 7). A finer angular /
               defocus grid, or optThr ~6.5, would refine these; finer search pays off
  weak 5-6     a little signal, below the gate
  none <5      no signal in the search there (template or CTF difference, or a marginal deposited peak)
Plus the pysmap detections that are not in the deposited list.

  cd $PYSMAP
  python examples/decolace_tric/lsu_gap.py --dest $WORK
Writes report/lsu_gap.png and report/lsu_gap.csv.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

CLASSES = [("found", "#1baf7a"), ("refined", "#2a78d6"), ("gate 6-7", "#eda100"),
           ("weak 5-6", "#eb6834"), ("none <5", "#6b6b6b")]


def main(argv=None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (smap)
    from smap.cutoff import load_z
    spec = importlib.util.spec_from_file_location("rev", Path(__file__).with_name("decolace_review.py"))
    rev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rev)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", required=True)
    ap.add_argument("--snr-deposited", type=float, default=7.75)
    ap.add_argument("--match-A", type=float, default=20.0)
    ap.add_argument("--results", default="results_LSU_control",
                    help="LSU search folder, e.g. results_LSU_control_a1.88 for the 1.88 deg run")
    a = ap.parse_args(argv)
    d = Path(a.dest).expanduser().resolve()
    tiles = {t["stem"]: t for t in rev.read_csv(d / "tiles.csv")}
    out = d / "report"
    out.mkdir(exist_ok=True)

    runs = []
    for f in sorted((d / a.results).glob("*")):
        if f.is_dir() and f.name in tiles and (f / "search_mip_scaled.mrc").exists():
            runs.append((f.name, f, tiles[f.name]))
    for f in (sorted((d / "results_controls").glob("LSU_*_filled")) if a.results == "results_LSU_control" else []):
        k = f.name.split("_")[1]
        stem = next((s for s in tiles if f"_{k}_" in s), None)
        if stem and (f / "search_mip_scaled.mrc").exists():
            runs.append((f"{k} filled", f, tiles[stem]))
    if not runs:
        raise SystemExit("no finished LSU search with search_mip_scaled.mrc")

    def key(stem):
        """Five-digit tile number in a file stem."""
        m = re.search(r"_(\d{5})_", stem)
        return m.group(1) if m else stem[:12]

    data = []
    import mrcfile
    for label, f, t in runs:
        with mrcfile.open(t["search_image"], permissive=True, header_only=True) as m:
            h, w = int(m.header.ny), int(m.header.nx)
        pix = float(t["pixel"])
        db = d / t["lamella"] / f"{t['lamella']}.db"
        dep = rev.deposited_peaks(db, int(t["asset"]), a.snr_deposited)
        xy, sc, thr = rev.ours(f, pix)
        cut = float(thr) if thr else 7.42
        data.append(dict(label=label if "filled" in label else key(f.name), f=f, h=h, w=w, pix=pix, dep=dep,
                         xy=xy, sc=sc, cut=cut))
    convs = ["A", "A_flip", "px", "px_flip"]
    conv = max(convs, key=lambda c: sum(len(rev.match(r["xy"][r["sc"] >= r["cut"]],
                                                      rev.to_A(r["dep"], c, r["pix"], r["h"]), a.match_A))
                                        for r in data if len(r["dep"])))

    rows, summary = [], []
    for r in data:
        f, pix = r["f"], r["pix"]
        z, _ = load_z(f)
        st = json.loads((f / "search_status.json").read_text())
        a_s = float(st.get("sampling_A") or pix)
        ratio = a_s / pix
        cs = np.array(z.shape) // 2 + 1
        co = np.array([r["h"], r["w"]]) // 2 + 1

        def grid(xyA):
            """Positions in A -> zero-based (row, col) on the search grid."""
            rr = (xyA[:, 1] / pix - co[0] + 1) / ratio + cs[0] - 1
            cc = (xyA[:, 0] / pix - co[1] + 1) / ratio + cs[1] - 1
            return np.round(rr).astype(int), np.round(cc).astype(int)

        # self-check of the mapping with the detected particles (row_search/col_search are 1-based)
        pr = rev.read_csv(f / "particles.csv")
        if pr:
            gr, gc = grid(r["xy"])
            err = np.median(np.hypot(gr + 1 - np.array([float(p["row_search"]) for p in pr]),
                                     gc + 1 - np.array([float(p["col_search"]) for p in pr])))
        else:
            err = float("nan")

        depA = rev.to_A(r["dep"], conv, pix, r["h"])
        zr, zc = grid(depA)

        def znear(i, j, wpx=2):
            """Largest search z-score within wpx pixels of (i, j)."""
            win = z[max(i - wpx, 0):i + wpx + 1, max(j - wpx, 0):j + wpx + 1]
            return float(win.max()) if win.size else np.nan
        zs = np.array([znear(i, j) for i, j in zip(zr, zc)])
        hit = r["sc"] >= r["cut"]
        found = {j for _, j, _ in rev.match(r["xy"][hit], depA, a.match_A)}
        any_ref = {j for _, j, _ in rev.match(r["xy"], depA, a.match_A)}
        cls = []
        for j in range(len(depA)):
            if j in found:
                c = "found"
            elif j in any_ref:
                c = "refined"
            elif zs[j] >= 6:
                c = "gate 6-7"
            elif zs[j] >= 5:
                c = "weak 5-6"
            else:
                c = "none <5"
            cls.append(c)
            rows.append(dict(tile=r["label"], x_A=round(depA[j, 0], 1), y_A=round(depA[j, 1], 1),
                             authors_snr=round(r["dep"][j, 2], 2), our_search_z=round(zs[j], 2), cls=c))
        ours_only = int(hit.sum()) - len(found)
        n = {c: cls.count(c) for c, _ in CLASSES}
        summary.append(dict(r, z=zs, cls=np.array(cls), n=n, ours_only=ours_only, err=err))
        print(f"tile {r['label']:>12s}: authors {len(depA):3d} | " + "  ".join(f"{c} {n[c]:2d}" for c, _ in CLASSES)
              + f" | not in deposited list {ours_only} (cutoff {r['cut']:.2f}; mapping check {err:.1f} px)")
        parts = []
        for c in sorted({round(r["cut"], 2), 7.6, 7.8, 8.0}):
            h = r["sc"] >= c
            f_c = len(rev.match(r["xy"][h], depA, a.match_A))
            parts.append(f"{c:.2f}: detected {int(h.sum())}, deposited found {f_c}, not deposited {int(h.sum()) - f_c}")
        print("      by final cutoff -> " + " | ".join(parts))
    tag_csv = "" if a.results == "results_LSU_control" else "_" + a.results.replace("results_LSU_control_", "")
    with open(out / f"lsu_gap{tag_csv}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(summary), figsize=(5.2 * len(summary), 4.6), squeeze=False)
    for axx, s in zip(ax[0], summary):
        for c, col in CLASSES:
            m = s["cls"] == c
            axx.scatter(s["dep"][m, 2], s["z"][m], s=18, color=col, label=f"{c} ({int(m.sum())})")
        axx.axhline(7.0, color="#6b6b6b", ls="--", lw=0.8)
        axx.text(axx.get_xlim()[1], 7.05, "refinement gate (optThr 7)", ha="right", va="bottom", fontsize=7,
                 color="#6b6b6b")
        axx.set_xlabel("authors' SNR (cisTEM)")
        axx.set_ylabel("search z at that position")
        axx.set_title(f"tile {s['label']}: {len(s['dep'])} deposited LSU", loc="left", fontsize=10)
        axx.legend(frameon=False, fontsize=7.5)
        axx.grid(color="#e6e6e3", lw=0.6)
    fig.tight_layout()
    tag = "" if a.results == "results_LSU_control" else "_" + a.results.replace("results_LSU_control_", "")
    fig.savefig(out / f"lsu_gap{tag}.png", dpi=150, facecolor="white")
    print(f"-> {out / f'lsu_gap{tag}.png'}, {out / f'lsu_gap{tag_csv}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
