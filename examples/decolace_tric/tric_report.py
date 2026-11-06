#!/usr/bin/env python
"""TRiC on DeCo-LACE: one figure and one table from what is finished in $WORK.

Reads (whatever exists): review/ (decolace_review.py output), results_TRiC/*/cutoffs.json
(smap.cutoff), results_controls/ (filled tile, phase-randomised tiles), results_TRiC_open/,
results_TRiC_decoy/ (decoy protein searched like TRiC, make_decoy_jobs.py).

  A  LSU control: the authors' ribosomes vs ours (and on the platinum-filled tile 520)
  B  per tile: clean closed-TRiC hits; expected from noise (each tile's own tail fit); measured
     on the phase-randomised tile; open-TRiC hits
  C  score distribution of the clean hits vs the noise expectation
  D  (when the decoy searches exist) target-decoy: TRiC vs decoy hits above each score on the same
     tiles, and the false-discovery estimate decoy / TRiC

  cd $PYSMAP
  python examples/decolace_tric/tric_report.py --dest $WORK
Writes report/tric_report.png and report/tric_report.csv.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

INK, INK2, GRID = "#2b2b2b", "#6b6b6b", "#e6e6e3"
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#c2185b"


def rows(p: Path) -> list:
    return list(csv.DictReader(open(p))) if p.exists() else []


def tile_key(stem: str) -> str:
    m = re.search(r"_(\d{5})_", stem)
    return m.group(1) if m else stem[:12]


def particles(folder: Path) -> tuple:
    """scores, (x_A, y_A) and the v2.1 cutoff of one search folder."""
    st = folder / "search_status.json"
    thr = json.loads(st.read_text()).get("gaussian_threshold") if st.exists() else None
    pr = rows(folder / "particles.csv")
    pix = None
    if st.exists():
        txt = json.loads(st.read_text()).get("provenance", {}).get("params_text", "")
        m = re.search(r"^aPerPix\s+([\d.]+)", txt, re.M)
        pix = float(m.group(1)) if m else None
    pix = pix or 1.5
    sc = np.array([float(r["score_final"]) for r in pr])
    xy = np.array([[float(r["x_orig0_est"]) * pix, float(r["y_orig0_est"]) * pix] for r in pr]).reshape(-1, 2)
    return sc, xy, thr


def noise_expected(folder: Path, cut: float) -> float:
    """Expected noise hits >= cut from the tile's own tail fit (1 at gpd_cluster)."""
    f = folder / "cutoffs.json"
    if not f.exists():
        return float("nan")
    c = json.loads(f.read_text())
    tail, c1 = c.get("tail") or {}, c.get("gpd_cluster")
    if not tail or c1 is None:
        return float("nan")
    u, xi, s = tail["u"], tail["shape"], tail["scale"]

    def surv(z):
        return np.exp(-(z - u) / s) if abs(xi) < 1e-6 else max(1 + xi * (z - u) / s, 1e-300) ** (-1 / xi)
    return float(surv(cut) / surv(c1))


def finished(folder: Path) -> bool:
    return (folder / ".done").exists() or (folder / "search_status.json").exists()


def clean_hits(folder: Path, cut: float, tile: dict | None, rev, sio) -> tuple:
    """Scores and positions (A) of the refined hits >= cut outside the artifact zones."""
    if not rows(folder / "particles.csv"):
        return np.zeros(0), np.zeros((0, 2))
    sc, xy, _ = particles(folder)
    flag = np.zeros(len(sc), bool)
    if tile and Path(tile["search_image"]).exists():
        art, b = rev.artifact_mask(sio.mr(tile["search_image"]).astype(np.float32), float(tile["pixel"]))
        flag = rev.in_mask(art, b, xy, float(tile["pixel"]))
    keep = (sc >= cut) & ~flag
    return sc[keep], xy[keep]


def match_count(a: np.ndarray, b: np.ndarray, r: float = 20.0) -> int:
    if not len(a) or not len(b):
        return 0
    D = np.linalg.norm(a[:, None] - b[None], axis=2)
    used, n = set(), 0
    for i in np.argsort(D.min(1)):
        j = int(np.argmin(np.where(np.isin(np.arange(len(b)), list(used)), np.inf, D[i])))
        if D[i, j] <= r and j not in used:
            used.add(j)
            n += 1
    return n


def main(argv=None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (smap)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", required=True)
    ap.add_argument("--tag", default="", help="e.g. opt6.5: the re-refined searches (review_opt6.5/, results_*_opt6.5/)")
    ap.add_argument("--strict", type=float, default=7.8,
                    help="second cutoff, set from the refined scores of the phase-randomised tiles")
    a = ap.parse_args(argv)
    d = Path(a.dest).expanduser().resolve()
    sfx = f"_{a.tag}" if a.tag else ""
    out = d / "report"
    out.mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location("rev", Path(__file__).with_name("decolace_review.py"))
    rev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rev)
    from smap import io as sio

    S = rows(d / f"review{sfx}" / "summary.csv")
    if not S:
        raise SystemExit(f"run decolace_review.py{' --tag ' + a.tag if a.tag else ''} first (review{sfx}/summary.csv missing)")
    dep = {}
    for r in rows(d / f"review{sfx}" / "deposited_lsu.csv"):
        dep.setdefault(r["stem"], []).append((float(r["x_A"]), float(r["y_A"])))
    dep = {k: np.array(v) for k, v in dep.items()}
    tiles = {t["stem"]: t for t in rows(d / "tiles.csv")}

    # ---- A: LSU control (+ filled tile)
    lsu = []
    for s in [s for s in S if s["target"] == "LSU_control"]:
        lsu.append(dict(label=f"tile {tile_key(s['stem'])}", authors=int(s["deposited"]), ours=int(s["detected"]),
                        same=int(s["matched"])))
    for f in sorted((d / "results_controls").glob("LSU_*_filled")):
        if not ((f / ".done").exists() or (f / "particles.csv").exists()):
            continue
        k = f.name.split("_")[1]
        stem = next((s for s in dep if f"_{k}_" in s), None)
        sc, xy, thr = particles(f)
        sel = xy[sc >= (thr or 7.42)]
        lsu.append(dict(label=f"tile {k}\nplatinum filled", authors=len(dep.get(stem, [])), ours=len(sel),
                        same=match_count(sel, dep.get(stem, np.zeros((0, 2))))))

    # ---- B/C: per tile
    det = {}
    for r in rows(d / f"review{sfx}" / "tric_detections.csv"):
        if r.get("artifact_zone", "0") == "0":
            det.setdefault(r["stem"], []).append(float(r["score"]))
    table, decoy = [], []
    open_dir = d / f"results_TRiC_open{sfx}"
    for s in [s for s in S if s["target"] == "TRiC"]:
        stem, k = s["stem"], tile_key(s["stem"])
        cut = float(s["cutoff"])
        row = dict(tile=k, clean=int(s["clean"]), artifact=int(s["in_artifact_zone"]),
                   noise_expected=round(noise_expected(d / f"results_TRiC{sfx}" / stem, cut), 2),
                   clean_strict=int(np.sum(np.array(det.get(stem, [])) >= a.strict)),
                   noise_measured="", noise_strict="", open_clean="", open_strict="", authors_lsu=int(s["deposited"]),
                   on_lsu=int(s["on_lsu_clean"]), chance_on_lsu=round(float(s["clean"]) * float(s["chance_on_lsu"]), 1))
        pr = sorted((d / f"results_controls{sfx}").glob(f"TRiC_{k}_phaserand*"))
        if pr and ((pr[0] / ".done").exists() or (pr[0] / "search_status.json").exists()):
            scn, _, thrn = particles(pr[0])
            row["noise_measured"] = int(np.sum(scn >= (thrn or cut)))
            row["noise_strict"] = int(np.sum(scn >= a.strict))
        of = open_dir / stem
        if (of / ".done").exists() or (of / "search_status.json").exists():
            sco, xyo, thro = particles(of)
            t = tiles.get(stem)
            if t and Path(t["search_image"]).exists():
                art, b = rev.artifact_mask(sio.mr(t["search_image"]).astype(np.float32), float(t["pixel"]))
                flag = rev.in_mask(art, b, xyo, float(t["pixel"]))
            else:
                flag = np.zeros(len(sco), bool)
            row["open_clean"] = int(np.sum((sco >= (thro or cut)) & ~flag))
            row["open_strict"] = int(np.sum((sco >= a.strict) & ~flag))
        row["decoy_clean"], row["decoy_strict"], row["decoy_on_tric"] = "", "", ""
        df = d / f"results_TRiC_decoy{sfx}" / stem
        if finished(df) and (df / "particles.csv").exists():
            # all refined hits (>= optThr) so the curves in D can start below the cutoff
            st_, xt = clean_hits(d / f"results_TRiC{sfx}" / stem, -np.inf, tiles.get(stem), rev, sio)
            sd_, xd = clean_hits(df, -np.inf, tiles.get(stem), rev, sio)
            row["decoy_clean"] = int(np.sum(sd_ >= cut))
            row["decoy_strict"] = int(np.sum(sd_ >= a.strict))
            row["decoy_on_tric"] = match_count(xd[sd_ >= cut], xt[st_ >= cut], r=50.0)
            decoy.append(dict(tile=k, cut=cut, tric=st_, decoy=sd_))
        table.append(row)
    table.sort(key=lambda r: r["tile"])
    with open(out / f"tric_report{sfx}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    clean_scores = np.array([float(r["score"]) for r in rows(d / f"review{sfx}" / "tric_detections.csv")
                             if r.get("artifact_zone", "0") == "0"])

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#9a9a96", "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    if decoy:
        fig, ax = plt.subplots(1, 4, figsize=(20, 4.8), gridspec_kw=dict(width_ratios=[0.9, 1.7, 1.2, 1.0]))
    else:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw=dict(width_ratios=[0.9, 1.7, 1.2]))
    A = ax[0]
    x = np.arange(len(lsu))
    w = 0.26
    for k_, (key, col, lab) in enumerate([("authors", INK2, "authors (cisTEM, SNR >= 7.75)"),
                                          ("ours", BLUE, "ours (pysmap, >= cutoff)"),
                                          ("same", AQUA, "same ribosome (<= 20 A)")]):
        v = [r[key] for r in lsu]
        bars = A.bar(x + (k_ - 1) * w, v, w * 0.92, color=col, label=lab)
        for b_, vv in zip(bars, v):
            A.text(b_.get_x() + b_.get_width() / 2, vv + 1, str(vv), ha="center", fontsize=8, color=INK)
    A.set_xticks(x, [r["label"] for r in lsu], fontsize=8)
    A.set_ylabel("ribosomes (LSU) per tile")
    A.set_title("A  LSU control", loc="left", fontsize=10, color=INK)
    A.legend(frameon=False, fontsize=7.5, loc="upper right")
    A.grid(axis="y", color=GRID, lw=0.6)
    A.set_axisbelow(True)
    A.set_ylim(0, max(max(r["authors"] for r in lsu), 10) * 1.25)

    B = ax[1]
    xb = np.arange(len(table))
    groups = [("clean", BLUE, "closed TRiC: hits outside artifact zones")]
    if any(r["open_clean"] != "" for r in table):
        groups.append(("open_clean", AQUA, "open TRiC: hits outside artifact zones"))
    if any(r["decoy_clean"] != "" for r in table):
        groups.append(("decoy_clean", RED, "decoy (lumazine synthase): hits outside artifact zones"))
    wb = 0.62 / len(groups) if len(groups) > 1 else 0.62
    cl = [r["clean"] for r in table]
    for g, (key, col, lab) in enumerate(groups):
        v = [r[key] if r[key] != "" else 0 for r in table]
        bars = B.bar(xb + (g - (len(groups) - 1) / 2) * wb, v, wb, color=col, label=lab)
        ks = key.replace("_clean", "_strict") if key != "clean" else "clean_strict"
        vs = [r[ks] if r.get(ks, "") != "" else 0 for r in table]
        B.bar(xb + (g - (len(groups) - 1) / 2) * wb, vs, wb, color="none", edgecolor=INK, hatch="////", lw=0.6,
              label=f"hatched: >= {a.strict:g}" if g == 0 else None)
        for b_, vv, r in zip(bars, v, table):
            if r[key] != "":
                B.text(b_.get_x() + b_.get_width() / 2, vv + 0.15, str(vv), ha="center", fontsize=7, color=INK)
    ne = [r["noise_expected"] for r in table]
    B.plot(xb, ne, "o-", color=ORANGE, lw=2, ms=5, label="noise, tail-fit model (search stage; underestimates)")
    nm = [(i, r["noise_measured"]) for i, r in enumerate(table) if r["noise_measured"] != ""]
    nms = [(i, r["noise_strict"]) for i, r in enumerate(table) if r["noise_strict"] != ""]
    if nms:
        B.plot([i for i, _ in nms], [v for _, v in nms], "D", color="white", ms=7, mec=INK, mew=1.0,
               label=f"noise, measured, >= {a.strict:g}", zorder=6)
    if nm:
        B.plot([i for i, _ in nm], [v for _, v in nm], "D", color=YELLOW, ms=8, mec=INK, mew=0.6,
               label="noise, measured (phase-randomised tile)", zorder=5)
    for i, r in enumerate(table):
        if r["artifact"] > 20:
            B.annotate(f"+{r['artifact']} on an\nimage artifact\n(excluded)", (i, r["clean"] + 0.5), xytext=(i - 1.8, max(cl) * 1.05),
                       fontsize=7.5, color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
    B.set_xticks(xb, [r["tile"] for r in table], rotation=45, fontsize=7.5)
    B.set_xlabel("tile")
    B.set_ylabel("detections per tile")
    B.set_ylim(0, max(cl) * 1.4)
    cls_ = sum(r["clean_strict"] for r in table)
    B.set_title(f"B  TRiC per tile: {sum(cl)} clean hits >= {cut:.2f}, {cls_} >= {a.strict:g} (hatched)",
                loc="left", fontsize=10, color=INK)
    B.legend(frameon=False, fontsize=7.5, loc="upper left")
    B.grid(axis="y", color=GRID, lw=0.6)
    B.set_axisbelow(True)

    C = ax[2]
    cut = min(float(s["cutoff"]) for s in S if s["target"] == "TRiC")
    edges = np.arange(np.floor(cut * 10) / 10, max(9.6, clean_scores.max() + 0.2 if len(clean_scores) else 9.6), 0.1)
    h, _ = np.histogram(clean_scores, edges)
    C.bar(edges[:-1] + 0.05, h, 0.085, color=BLUE, label=f"clean closed-TRiC hits (n = {len(clean_scores)})")
    zs = np.linspace(cut, edges[-1], 200)
    exp_tot = np.nansum(ne)
    exp_curve = []
    for z in zs:
        exp_curve.append(np.nansum([noise_expected(d / f"results_TRiC{sfx}" / s["stem"], z) for s in S if s["target"] == "TRiC"]))
    exp_curve = np.array(exp_curve)
    dens = -np.gradient(exp_curve, zs) * 0.1
    C.plot(zs, dens, color=ORANGE, lw=2, label=f"tail-fit noise model, search stage (total {exp_tot:.1f})")
    C.set_xlabel("final score (z)")
    C.set_ylabel("hits per 0.1 z")
    C.set_title("C  Scores of the clean hits", loc="left", fontsize=10, color=INK)
    C.legend(frameon=False, fontsize=7.5, loc="upper right")
    C.grid(axis="y", color=GRID, lw=0.6)
    C.set_axisbelow(True)
    if decoy:
        Dx = ax[3]
        st_all = np.concatenate([p["tric"] for p in decoy])
        sd_all = np.concatenate([p["decoy"] for p in decoy])
        cut_d = min(p["cut"] for p in decoy)
        zs = np.linspace(7.0, max(9.6, np.max(np.r_[st_all, sd_all, 0]) + 0.2), 300)
        nt = np.array([np.sum(st_all >= z) for z in zs])
        nd = np.array([np.sum(sd_all >= z) for z in zs])
        Dx.step(zs, nt, where="post", color=BLUE, lw=2, label="TRiC (closed)")
        Dx.step(zs, nd, where="post", color=RED, lw=2, label="decoy (lumazine synthase)")
        Dx.axvline(cut_d, color=INK2, lw=0.8, ls="--")
        Dx.axvline(a.strict, color=INK, lw=0.8, ls=":")
        n_t = sum(int(np.sum(p["tric"] >= p["cut"])) for p in decoy)
        n_d = sum(int(np.sum(p["decoy"] >= p["cut"])) for p in decoy)
        Dx.text(0.97, 0.62, f"at each tile's cutoff:\nTRiC {n_t}, decoy {n_d}\n"
                f"false-discovery estimate {n_d / max(n_t, 1):.0%}", transform=Dx.transAxes,
                ha="right", fontsize=8, color=INK)
        Dx.set_xlabel("final score (z)")
        Dx.set_ylabel("hits with score >= z")
        Dx.set_title(f"D  Target vs decoy ({', '.join(p['tile'] for p in decoy)})", loc="left",
                     fontsize=10, color=INK)
        Dx.legend(frameon=False, fontsize=7.5, loc="upper right")
        Dx.grid(color=GRID, lw=0.6)
        Dx.set_axisbelow(True)
    on = sum(r["on_lsu"] for r in table)
    ch = sum(r["chance_on_lsu"] for r in table)
    fig.text(0.01, -0.03, f"DeCo-LACE lamellae (EMPIAR-11063). Clean hits on an authors' ribosome: {on} (chance {ch:.1f}). "
             "Noise expectation from each tile's z-score tail (smap.cutoff); measured noise from phase-randomised copies of the tile.",
             fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(out / f"tric_report{sfx}.png", dpi=180, bbox_inches="tight", facecolor="white")
    def two(r, k1, k2):
        return f"{r[k1]}/{r[k2]}" if r[k1] != "" else ""
    print(f"counts as  >= {cut:.2f} / >= {a.strict:g}")
    print(f"{'tile':>6s} {'TRiC':>7s} {'noise meas':>10s} {'open':>7s} {'decoy':>7s} {'on LSU':>6s} {'chance':>6s}")
    for r in table:
        print(f"{r['tile']:>6s} {two(r, 'clean', 'clean_strict'):>7s} {two(r, 'noise_measured', 'noise_strict'):>10s} "
              f"{two(r, 'open_clean', 'open_strict'):>7s} {two(r, 'decoy_clean', 'decoy_strict'):>7s} "
              f"{r['on_lsu']:>6d} {r['chance_on_lsu']:>6}")
    tot = lambda k: sum(r[k] for r in table if r[k] != "") if any(r[k] != "" for r in table) else "-"  # noqa: E731
    print(f"{'total':>6s} {str(tot('clean')) + '/' + str(tot('clean_strict')):>7s} "
          f"{str(tot('noise_measured')) + '/' + str(tot('noise_strict')):>10s} "
          f"{str(tot('open_clean')) + '/' + str(tot('open_strict')):>7s} "
          f"{str(tot('decoy_clean')) + '/' + str(tot('decoy_strict')):>7s}   (open, decoy, noise: only the tiles done)")
    for p in decoy:
        a1, a2 = p["tric"] >= p["cut"], p["decoy"] >= p["cut"]
        r = next(r for r in table if r["tile"] == p["tile"])
        print(f"target-decoy tile {p['tile']} (cutoff {p['cut']:.2f}): TRiC {a1.sum()}, decoy {a2.sum()} "
              f"({r['decoy_on_tric']} of them within 50 A of a TRiC hit)")
    for r in lsu:
        print(f"LSU {r['label'].replace(chr(10), ' ')}: authors {r['authors']}, ours {r['ours']}, same {r['same']}")
    print(f"-> {out / f'tric_report{sfx}.png'}, {out / f'tric_report{sfx}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
