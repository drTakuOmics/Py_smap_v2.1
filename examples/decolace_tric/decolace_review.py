#!/usr/bin/env python
"""Review of the DeCo-LACE runs: LSU control vs the authors' deposited 2DTM peaks, and TRiC
detections vs the ribosome positions.

For every finished tile (``.done``):
  LSU control: pysmap detections (final score >= cutoff) matched one-to-one to the authors' LSU
               peaks (SNR >= 7.75, 6SWA, cisTEM) within --match-A. The coordinate convention of
               the deposited peaks (Å or pixels; y up or down) is chosen as the one with the
               most matches over all control tiles and reported.
  TRiC:        detections above the cutoff, each with the distance to the nearest deposited LSU
               peak. ``on_lsu`` = closer than --overlap-A (projections overlap). The fraction on
               ribosomes is compared with the fraction expected for random positions inside the
               illuminated area (mask): an excess means the TRiC template also scores on
               ribosomes (cross-matching), not TRiC.

Cutoff per tile: --cutoff, or "erfc" (the v2.1 Gaussian threshold of that search, default).

  cd $PYSMAP
  python examples/decolace_tric/decolace_review.py --dest $WORK
Outputs in $WORK/review/ and $WORK_review_small.tgz (tables only, for sharing).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
import tarfile
from pathlib import Path

import numpy as np


def read_csv(p: Path) -> list:
    """Rows of a CSV file ([] if it does not exist)."""
    return list(csv.DictReader(open(p))) if p.exists() else []


def deposited_peaks(db: Path, asset: int, snr_min: float) -> np.ndarray:
    """(n, 3) x, y, snr of the authors' LSU peaks for one image (the match job with most peaks)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    tabs = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
    cols = {t: [r[1] for r in con.execute(f"pragma table_info('{t}')")] for t in tabs}
    mapt = next(t for t in tabs if {"TEMPLATE_MATCH_ID", "IMAGE_ASSET_ID"} <= set(cols[t]))
    ids = [r[0] for r in con.execute(f"select TEMPLATE_MATCH_ID from {mapt} where IMAGE_ASSET_ID=?", (asset,))]
    best = np.zeros((0, 3))
    for tid in ids:
        t = next((t for t in tabs if re.search(rf"PEAK.*_{tid}$", t) and "PEAK_HEIGHT" in cols[t]), None)
        if not t or not {"X_POSITION", "Y_POSITION"} <= set(cols[t]):
            continue
        a = np.array(con.execute(f"select X_POSITION, Y_POSITION, PEAK_HEIGHT from {t} "
                                 f"where PEAK_HEIGHT >= ?", (snr_min,)).fetchall(), dtype=float)
        if len(a) > len(best):
            best = a.reshape(-1, 3)
    con.close()
    return best


def to_A(peaks: np.ndarray, conv: str, pixel: float, height_px: int) -> np.ndarray:
    """Deposited (x, y) -> Å in the image frame (x = column, y = row from the first row of the file)."""
    x, y = peaks[:, 0].copy(), peaks[:, 1].copy()
    if conv.startswith("px"):
        x, y = x * pixel, y * pixel
    if conv.endswith("flip"):
        y = (height_px - 1) * pixel - y
    return np.stack([x, y], 1)


def match(a: np.ndarray, b: np.ndarray, r: float) -> list:
    """Greedy one-to-one pairs (i, j, d) closer than r, nearest first."""
    if not len(a) or not len(b):
        return []
    D = np.linalg.norm(a[:, None] - b[None], axis=2)
    pairs, used_a, used_b = [], set(), set()
    for k in np.argsort(D, axis=None):
        i, j = np.unravel_index(k, D.shape)
        if D[i, j] > r:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((int(i), int(j), float(D[i, j])))
    return pairs


def ours(res: Path, pixel: float) -> tuple:
    """Detections of one search: positions (A), final scores and the search's own cutoff.
    The rotations are left in ours.R."""
    rows = read_csv(res / "particles.csv")
    st = json.loads((res / "search_status.json").read_text()) if (res / "search_status.json").exists() else {}
    xy = np.array([[float(r["x_orig0_est"]) * pixel, float(r["y_orig0_est"]) * pixel] for r in rows]).reshape(-1, 2)
    sc = np.array([float(r["score_final"]) for r in rows])
    ours.R = [np.array([[float(r[f"q_opt_{i}{j}"]) for j in range(3)] for i in range(3)]) for r in rows]
    return xy, sc, st.get("gaussian_threshold")


def image_info(dest: Path, t: dict):
    """Height and width of a tile's search image and its illumination mask (None without one).
    The image itself is left in image_info.im."""
    from smap import io as sio
    img = Path(t["search_image"])
    im = sio.mr(img).astype(np.float32) if img.exists() else None
    h, w = im.shape if im is not None else (0, 0)
    image_info.im = im
    mask = None
    if t.get("mask_url"):
        mp = dest / t["lamella"] / "Images" / Path(t["mask_url"]).name
        if mp.exists():
            mask = sio.mr(mp) > 0.5
    return h, w, mask


def chance_fraction(lsu_A: np.ndarray, mask, h: int, w: int, pixel: float, r: float, n: int = 4000,
                    seed: int = 0) -> float:
    """Fraction of random positions (inside the mask, when given) that lie within r of a
    ribosome: the overlap expected by chance."""
    if not len(lsu_A) or h == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    if mask is not None and mask.shape == (h, w) and mask.any():
        yy, xx = np.nonzero(mask)
        k = rng.integers(0, len(yy), n)
        pts = np.stack([xx[k], yy[k]], 1) * pixel
    else:
        pts = np.stack([rng.uniform(0, w, n), rng.uniform(0, h, n)], 1) * pixel
    d = np.linalg.norm(pts[:, None] - lsu_A[None], axis=2).min(1)
    return float((d < r).mean())


def artifact_mask(img: np.ndarray, pixel: float, k: float = 5.0, margin_A: float = 200.0, bin_A: float = 12.0):
    """Strong low-resolution features (protective metal layer, lamella or crack edges, ice
    contamination, dense granules): the 18 Å-smoothed image deviating from its median by more than
    k robust SD, or its gradient above k robust SD, dilated by margin_A. Returns (binned mask, bin)."""
    from scipy.ndimage import binary_dilation, gaussian_filter, label, sobel
    b = max(1, int(round(bin_A / pixel)))
    n0, n1 = (img.shape[0] // b) * b, (img.shape[1] // b) * b
    L = gaussian_filter(img[:n0, :n1].reshape(n0 // b, b, n1 // b, b).mean(axis=(1, 3)), 1.5)

    def robust(x):
        """Median and robust SD (1.4826 x MAD)."""
        med = np.median(x)
        return med, 1.4826 * np.median(np.abs(x - med)) + 1e-12
    med, mad = robust(L)
    g = np.hypot(sobel(L, 0), sobel(L, 1))
    gmed, gmad = robust(g)
    art = (np.abs(L - med) > k * mad) | (g > gmed + k * gmad)
    lab, nlab = label(art)
    if nlab:
        sizes = np.bincount(lab.ravel())
        art = np.isin(lab, np.nonzero(sizes >= 4)[0][1:]) if len(sizes) > 1 else art
    it = int(round(margin_A / (b * pixel)))
    if it > 0 and art.any():
        art = binary_dilation(art, iterations=it)
    return art, b


def in_mask(art: np.ndarray, b: int, xy_A: np.ndarray, pixel: float) -> np.ndarray:
    """For each position (A): True if it falls in the artifact mask (binned by b pixels)."""
    if not len(xy_A):
        return np.zeros(0, bool)
    c = np.clip((xy_A[:, 0] / pixel / b).astype(int), 0, art.shape[1] - 1)
    r = np.clip((xy_A[:, 1] / pixel / b).astype(int), 0, art.shape[0] - 1)
    return art[r, c]


def _axial(Rs: np.ndarray) -> float:
    """Largest eigenvalue over the three model axes of the axial orientation tensor mean(u u^T),
    u = R e_k in the image frame (1/3 for uniform, 1 for one common axis; sign-invariant, so a
    top/bottom-symmetric barrel counts as aligned)."""
    best = 0.0
    for k in range(3):
        u = Rs[:, :, k]
        T = np.einsum("ni,nj->ij", u, u) / len(u)
        best = max(best, float(np.linalg.eigvalsh(T)[-1]))
    return best


def orientation_coherence(Rs: list, n_sim: int = 2000, seed: int = 0) -> tuple:
    """Axis alignment of the poses (``_axial``) and its p-value against uniform random rotations."""
    n = len(Rs)
    if n < 3:
        return float("nan"), float("nan")
    obs = _axial(np.asarray(Rs))
    rng = np.random.default_rng(seed)
    sim = []
    for _ in range(n_sim):
        q = rng.normal(size=(n, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        w, x, y, z = q.T
        R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                      np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                      np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)
        sim.append(_axial(R))
    return obs, float((np.sum(np.array(sim) >= obs) + 1) / (n_sim + 1))


def main(argv=None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (smap)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", required=True)
    ap.add_argument("--cutoff", default="erfc", help='"erfc" (per search) or a number')
    ap.add_argument("--snr-deposited", type=float, default=7.75)
    ap.add_argument("--match-A", type=float, default=20.0)
    ap.add_argument("--overlap-A", type=float, default=150.0)
    ap.add_argument("--artifact-k", type=float, default=5.0, help="robust SD for the artifact mask")
    ap.add_argument("--artifact-margin-A", type=float, default=200.0)
    ap.add_argument("--tag", default="", help="e.g. opt6.5: read results_TRiC_opt6.5, write review_opt6.5/")
    ap.add_argument("--lsu", default="results_LSU_control", help="ribosome search folder, e.g. results_LSU_mouse_opt6.5")
    a = ap.parse_args(argv)
    sfx = f"_{a.tag}" if a.tag else ""
    tric_dir, lsu_dir = f"results_TRiC{sfx}", a.lsu
    dest = Path(a.dest).expanduser().resolve()
    tiles = read_csv(dest / "tiles.csv")
    out = dest / f"review{sfx}"
    out.mkdir(exist_ok=True)

    def cut(thr):
        """Cutoff of one search: its own threshold with --cutoff erfc, otherwise the number given."""
        return float(thr) if a.cutoff == "erfc" and thr else float(a.cutoff if a.cutoff != "erfc" else 7.5)

    # ---- LSU control: pick the coordinate convention, then match
    ctrl = [t for t in tiles if (dest / lsu_dir / t["stem"] / ".done").exists()]
    data = {}
    for t in tiles:
        db = dest / t["lamella"] / f"{t['lamella']}.db"
        dep = deposited_peaks(db, int(t["asset"]), a.snr_deposited) if db.exists() else np.zeros((0, 3))
        if not any((dest / sub / t["stem"] / ".done").exists() for sub in (lsu_dir, tric_dir)):
            continue
        h, w, mask = image_info(dest, t)
        art = artifact_mask(image_info.im, float(t["pixel"]), a.artifact_k, a.artifact_margin_A) \
            if image_info.im is not None else (np.zeros((1, 1), bool), 1)
        data[t["stem"]] = (dep, h, w, mask, float(t["pixel"]), art)
    convs = ["A", "A_flip", "px", "px_flip"]
    score = {c: 0 for c in convs}
    for t in ctrl:
        dep, h, w, _, pix, _ = data[t["stem"]]
        xy, sc, thr = ours(dest / lsu_dir / t["stem"], pix)
        sel = sc >= cut(thr)
        for c in convs:
            score[c] += len(match(xy[sel], to_A(dep, c, pix, h), a.match_A)) if len(dep) else 0
    conv = max(convs, key=lambda c: score[c]) if ctrl else "A"
    print(f"deposited peak coordinates: '{conv}' (matches per convention {score})")

    summary, lsu_rows, tric_rows = [], [], []
    for t in ctrl:
        dep, h, w, _, pix, _ = data[t["stem"]]
        xy, sc, thr = ours(dest / lsu_dir / t["stem"], pix)
        c = cut(thr)
        sel = np.nonzero(sc >= c)[0]
        depA = to_A(dep, conv, pix, h)
        pr = match(xy[sel], depA, a.match_A)
        for i, j, d in pr:
            lsu_rows.append(dict(stem=t["stem"], ours_score=round(sc[sel[i]], 3), deposited_snr=dep[j, 2], dist_A=round(d, 1)))
        summary.append(dict(stem=t["stem"], target="LSU_control", cutoff=round(c, 3), refined=len(sc),
                            detected=len(sel), deposited=len(dep), matched=len(pr),
                            frac_deposited_found=round(len(pr) / len(dep), 3) if len(dep) else "",
                            top=round(float(sc.max()), 2) if len(sc) else ""))
        print(f"LSU {t['stem']}: ours {len(sel)} >= {c:.2f} (of {len(sc)} refined), deposited {len(dep)} "
              f">= {a.snr_deposited}, matched {len(pr)}")

    dep_rows = []
    for t in tiles:
        if t["stem"] in data:
            dep, h, w, mask, pix, (art, b) = data[t["stem"]]
            for x, y in to_A(dep, conv, pix, h):
                dep_rows.append(dict(stem=t["stem"], x_A=round(x, 1), y_A=round(y, 1)))
    for t in tiles:
        res = dest / tric_dir / t["stem"]
        if not (res / ".done").exists():
            continue
        dep, h, w, mask, pix, (art, b) = data[t["stem"]]
        xy, sc, thr = ours(res, pix)
        Rs = ours.R
        c = cut(thr)
        depA = to_A(dep, conv, pix, h)
        if mask is not None and len(depA) and mask.shape == (h, w):
            if in_mask(mask, 1, depA, pix).mean() < 0.5:        # the deposited mask marks the dark area
                mask = ~mask
        sel = np.nonzero(sc >= c)[0]
        dmin = (np.linalg.norm(xy[sel][:, None] - depA[None], axis=2).min(1)
                if len(sel) and len(depA) else np.full(len(sel), np.nan))
        flag = in_mask(art, b, xy[sel], pix)
        clean = ~flag
        region = mask if mask is not None and mask.shape == (h, w) else np.ones((h, w), bool)
        if region.shape == (h, w):
            up = np.kron(art, np.ones((b, b), bool))
            region = region.copy()
            region[:up.shape[0], :up.shape[1]] &= ~up
        ch = chance_fraction(depA, region, h, w, pix, a.overlap_A)
        coh, coh_p = orientation_coherence([Rs[i] for i in sel])
        coh_c, coh_cp = orientation_coherence([Rs[i] for i, f in zip(sel, flag) if not f])
        for k, i in enumerate(sel):
            tric_rows.append(dict(stem=t["stem"], num_id=int(i) + 1, score=round(sc[i], 3),
                                  x_A=round(xy[i, 0], 1), y_A=round(xy[i, 1], 1),
                                  nearest_lsu_A=round(float(dmin[k]), 1), on_lsu=int(dmin[k] < a.overlap_A),
                                  artifact_zone=int(flag[k])))
        summary.append(dict(stem=t["stem"], target="TRiC", cutoff=round(c, 3), refined=len(sc), detected=len(sel),
                            artifact_area=round(float(art.mean()), 3), in_artifact_zone=int(flag.sum()),
                            clean=int(clean.sum()), deposited=len(dep),
                            on_lsu_clean=int(np.sum(dmin[clean] < a.overlap_A)), chance_on_lsu=round(ch, 3),
                            orient_coherence=round(coh, 3), coherence_p=round(coh_p, 4),
                            orient_coherence_clean=round(coh_c, 3), coherence_p_clean=round(coh_cp, 4),
                            top_clean=round(float(sc[sel][clean].max()), 2) if clean.any() else ""))
    if tric_rows or summary:
        keys = sorted({k for r in summary for k in r}, key=lambda k: list(summary[0]).index(k) if k in summary[0] else 99)
        with open(out / "summary.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            wr.writerows(summary)
    for name, rows in (("lsu_control_matches.csv", lsu_rows), ("tric_detections.csv", tric_rows),
                       ("deposited_lsu.csv", dep_rows)):
        if rows:
            with open(out / name, "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
    tr = [s for s in summary if s["target"] == "TRiC"]
    if tr:
        for s_ in tr:
            print(f"TRiC {s_['stem'][13:24]}: {s_['detected']:3d} >= {s_['cutoff']:.2f}, {s_['in_artifact_zone']:3d} in "
                  f"artifact zone ({100 * s_['artifact_area']:.0f}% of image), {s_['clean']:2d} clean; clean on a "
                  f"ribosome {s_['on_lsu_clean']} (chance {s_['chance_on_lsu']:.2f}/detection); pose coherence "
                  f"{s_['orient_coherence']:.2f} (p {s_['coherence_p']:.3f}), clean {s_['orient_coherence_clean']:.2f} "
                  f"(p {s_['coherence_p_clean']:.3f})")
        n_c = sum(s_["clean"] for s_ in tr)
        n_on = sum(s_["on_lsu_clean"] for s_ in tr)
        exp = np.nansum([s_["clean"] * s_["chance_on_lsu"] for s_ in tr])
        print(f"TRiC total: {sum(s_['detected'] for s_ in tr)} detections, {n_c} outside artifact zones; "
              f"on a ribosome {n_on} (expected by chance {exp:.1f})")
    rep = dict(convention=conv, convention_scores=score, cutoff=a.cutoff, match_A=a.match_A,
               overlap_A=a.overlap_A, summary=summary)
    (out / "review.json").write_text(json.dumps(rep, indent=1, default=float))

    tgz = dest / f"review{sfx}_small.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        for f in out.iterdir():
            tf.add(f, arcname=f"review/{f.name}")
        tf.add(dest / "tiles.csv", arcname="tiles.csv")
        for f in sorted((dest / "report").glob("*.csv")) if (dest / "report").is_dir() else []:
            tf.add(f, arcname=f"report/{f.name}")
        subs = [lsu_dir, tric_dir] + [s for s in (f"results_controls{sfx}", f"results_TRiC_decoy{sfx}",
                                                 f"results_TRiC_open{sfx}") if (dest / s).is_dir()]
        for sub in dict.fromkeys(subs):
            for d in sorted((dest / sub).glob("*/")):
                if not (d / ".done").exists():
                    continue
                for n in ("particles.csv", "search_status.json", "cutoffs.json", "particles_qc.csv"):
                    if (d / n).exists():
                        tf.add(d / n, arcname=f"{sub}/{d.name}/{n}")
    print(f"tables in {out}; for sharing: {tgz} ({tgz.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
