#!/usr/bin/env python
"""Detections of many searches as one RELION particle STAR file (for relion_reconstruct or a
cryoSPARC particle import), with duplicates removed, crowded micrographs left out and, optionally,
the orientations compared with a reference particle list.

    python tools/export_detections.py RESULTS/*/ --out-dir EXPORT --diameter-A 250 [--min-score 7]
        [--reference-dir COORDS] [--max-overlap 0.5]

Per micrograph (detections with final score >= --min-score, default: each search's cutoff):

* duplicates: a particle detected twice at nearly the same position with two orientations (the
  clustering keeps candidates more than qThr apart in orientation, which symmetric particles often
  are). The best-scoring detection within 0.16 x --diameter-A is kept.
* overlap: pairs of remaining detections closer than 0.8 x --diameter-A, i.e. particles overlapping in
  projection (several layers). Micrographs with more than --max-overlap such pairs per detection are
  left out of the export (their particle images contain their neighbours).
* with --reference-dir (lists <folder name>.csv with x_px, y_px, rot, tilt, psi, as written by
  tools/star_to_pars.py): detections paired with reference particles (within 0.16 x --diameter-A)
  and the angle between the two orientations, the smallest over the symmetry operators of the search
  (symmetryFile). RELION angles are the ZYZ decomposition of the search rotation (smap.export).

Particle images: the centred patches of each search's particles.mrc, cut from the micrograph as
recorded (raw values: a dose-weighted sum has a mean of tens of counts per pixel and a much smaller
SD). Each exported image is normalised (background outside --bg-radius, default 0.75 of the box
radius and at least the particle radius: mean 0, SD 1), as RELION and cryoSPARC expect: a large mean
becomes a bright disk under the real-space window of the reconstruction and a featureless ball in
the map. Check the export with tools/check_reconstruction.py before cryoSPARC or RELION.

Writes <out-dir>/particles.star (RELION 3.1: optics + particles; rlnImageName j@stacks/<micrograph>.mrcs;
rlnRandomSubset 1/2, two random half-sets, which cryoSPARC needs to import the orientations),
<out-dir>/stacks/ (one normalised stack per micrograph, float32), <out-dir>/micrographs.csv (one row
per micrograph, with the raw image mean and background SD) and, with a reference, <out-dir>/poses.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # repo root (smap)
sys.path.insert(0, str(Path(__file__).resolve().parent))         # tools/


def par_value(text: str, key: str):
    """Values of one key in the text of a .par file (None if the key is absent)."""
    m = re.search(rf"^{key}\s+(.+)$", text, re.M)
    return m.group(1).split() if m else None


def dedupe(xy: np.ndarray, score: np.ndarray, r: float) -> np.ndarray:
    """Indices kept: best score first, nothing kept within r of a kept one."""
    keep = []
    for i in np.argsort(-score):
        if all(np.hypot(*(xy[i] - xy[j])) >= r for j in keep):
            keep.append(i)
    return np.array(sorted(keep), dtype=int)


def normalise(img: np.ndarray, r_px: float) -> tuple:
    """(image with background mean 0 and SD 1, raw mean, raw background SD); background = outside r_px."""
    n = img.shape[0]
    g = np.arange(n) - n // 2
    bg = np.hypot(*np.meshgrid(g, g, indexing="ij")) > r_px
    if bg.sum() < 0.05 * img.size:
        bg = np.ones_like(bg)
    mu, sd = float(img[bg].mean()), float(img[bg].std())
    return (img - mu) / (sd if sd > 0 else 1.0), float(img.mean()), sd


def close_pairs(xy: np.ndarray, r: float) -> int:
    """Number of pairs of positions closer than r."""
    if len(xy) < 2:
        return 0
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    return int((d[np.triu_indices(len(xy), 1)] < r).sum())


def angle_deg(A: np.ndarray, B: np.ndarray) -> float:
    """Rotation angle between two rotation matrices."""
    c = (np.trace(A.T @ B) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def sym_angle(R: np.ndarray, A: np.ndarray, S) -> float:
    """Smallest angle between R and A S^T over the operators S (same template for R and R S^T)."""
    return min(angle_deg(R, A @ M.T) for M in S)


def main(argv=None) -> int:
    from match_positions import pair
    from smap import io as sio
    from smap.export import relion_matrix, rows_from_smap, write_star

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folders", nargs="+", help="search output folders")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--diameter-A", type=float, required=True, help="particle diameter")
    ap.add_argument("--min-score", type=float, default=None, help="default: each search's cutoff")
    ap.add_argument("--max-overlap", type=float, default=0.5,
                    help="leave out micrographs with more overlapping pairs per detection than this")
    ap.add_argument("--reference-dir", default=None, help="reference lists <folder name>.csv (with angles)")
    ap.add_argument("--amp-contrast", type=float, default=0.07)
    ap.add_argument("--bg-radius-A", type=float, default=None,
                    help="normalisation background outside this radius (default: 0.75 of the box radius, at least "
                         "the particle radius)")
    a = ap.parse_args(argv)
    out = Path(a.out_dir).expanduser()
    (out / "stacks").mkdir(parents=True, exist_ok=True)
    r_dup, r_ovl = 0.16 * a.diameter_A, 0.8 * a.diameter_A          # same particle / overlapping in projection (A)
    rows_all, table, poses, optics, box = [], [], [], None, None
    S_cache = {}                                                     # symmetry operators per symmetryFile
    for f in map(Path, a.folders):
        if not (f / "particles.csv").exists():
            print(f"{f.name}: no particles.csv, skipped")
            continue
        # ---- settings of this search (the .par text is stored with the results); one optics group overall
        st = json.loads((f / "search_status.json").read_text())
        txt = st.get("provenance", {}).get("params_text", "")
        pix = float(par_value(txt, "aPerPix")[0])
        opt = (pix, float(par_value(txt, "V_acc")[0]) / 1000, float(par_value(txt, "Cs")[0]) * 1000)
        if optics is None:
            optics = opt
        elif any(abs(u - v) > 1e-3 for u, v in zip(opt, optics)):
            print(f"{f.name}: optics {opt} differ from {optics}; skipped (one optics group)")
            continue
        # ---- detections at the cutoff; duplicates removed; overlap counted on what is left
        img = Path((par_value(txt, "imageFile") or [f.name])[0]).name
        cut = a.min_score if a.min_score is not None else float(st.get("gaussian_threshold") or 7.0)
        rows = rows_from_smap(f, img, images=True)
        rows = [r for r in rows if r["score"] >= cut]
        xy = np.array([[r["x"], r["y"]] for r in rows]).reshape(-1, 2)
        score = np.array([r["score"] for r in rows])
        keep = dedupe(xy, score, r_dup / pix)
        n_ovl = close_pairs(xy[keep], r_ovl / pix)
        frac = n_ovl / max(len(keep), 1)
        excluded = frac > a.max_overlap
        rec = dict(micrograph=f.name, cutoff=round(cut, 2), detected=len(rows), duplicates=len(rows) - len(keep),
                   kept=len(keep), overlap_pairs=n_ovl, overlap_per_detection=round(frac, 3),
                   exported=0 if excluded else len(keep))
        # ---- particle images: the kept patches, normalised, as one stack per micrograph
        stack = out / "stacks" / f"{f.name}.mrcs"
        if not excluded and len(keep):
            import mrcfile
            with mrcfile.mmap(f / "particles.mrc", permissive=True, mode="r") as m:
                src = m.data if m.data.ndim == 3 else m.data[None]
                n_box = int(src.shape[-1])
                box = n_box if box is None else box
                r_bg = (a.bg_radius_A / pix) if a.bg_radius_A else max(0.75 * n_box / 2, 0.5 * a.diameter_A / pix)
                norm = [normalise(np.asarray(src[int(rows[i]["image"].split("@")[0]) - 1], np.float64), r_bg)
                        for i in keep]
            if stack.is_symlink() or stack.exists():
                stack.unlink()          # e.g. a link to particles.mrc from an earlier export: never write through it
            with mrcfile.new(stack) as m:
                m.set_data(np.stack([x for x, _, _ in norm]).astype(np.float32))
                m.set_image_stack()
                m.voxel_size = pix
            for k, i in enumerate(keep, 1):
                r = dict(rows[i])
                r["image"] = f"{k:06d}@stacks/{stack.name}"
                rows_all.append(r)
            rec.update(raw_mean=round(float(np.median([u for _, u, _ in norm])), 4),
                       raw_background_sd=round(float(np.median([v for _, _, v in norm])), 4))
        # ---- orientations against the reference list (also for micrographs left out of the export)
        if a.reference_dir:
            ref_f = Path(a.reference_dir) / f"{f.name}.csv"
            if ref_f.exists():
                ref = list(csv.DictReader(open(ref_f)))
                rxy = np.array([[float(r["x_px"]), float(r["y_px"])] for r in ref]).reshape(-1, 2)
                sym = par_value(txt, "symmetryFile")
                key = sym[0] if sym else None
                if key not in S_cache:
                    S_cache[key] = sio.read_rotations_file(key) if key else np.eye(3)[None]
                pr = pair(xy[keep], rxy, r_dup / pix)
                angs = []
                for i, j in pr:
                    rr = ref[j]
                    A = relion_matrix(float(rr["rot"]), float(rr["tilt"]), float(rr["psi"]))
                    ang = sym_angle(rows[keep[i]]["R"], A, S_cache[key])
                    angs.append(ang)
                    poses.append(dict(micrograph=f.name, num_id=rows[keep[i]]["image"].split("@")[0],
                                      score=round(rows[keep[i]]["score"], 3), reference_index=j,
                                      angle_deg=round(ang, 3)))
                rec.update(paired=len(pr), angle_median_deg=round(float(np.median(angs)), 3) if angs else "")
        table.append(rec)

    if not table:
        print("nothing to export")
        return 1
    # ---- outputs: per-micrograph table, STAR file with two random half-sets, pose comparison
    keys = list(dict.fromkeys(k for r in table for k in r))
    with open(out / "micrographs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(table)
    perm = np.random.default_rng(0).permutation(len(rows_all))          # two equal random half-sets
    for k, i in enumerate(perm):
        rows_all[i]["subset"] = 1 + k % 2
    n = write_star(rows_all, out / "particles.star", optics[0], optics[1], optics[2], a.amp_contrast, 0, box) \
        if rows_all else 0
    det = sum(r["detected"] for r in table)
    dup = sum(r["duplicates"] for r in table)
    exc = [r for r in table if r["exported"] == 0 and r["kept"]]
    print(f"{len(table)} micrographs: {det} detections, {dup} duplicates removed, {det - dup} particles; "
          f"{len(exc)} micrographs left out (overlap > {a.max_overlap:g} pairs per detection: "
          f"{', '.join(r['micrograph'][:38] for r in exc) or 'none'})")
    print(f"exported {n} particles ({box} px boxes, {optics[0]:g} A/px) -> {out / 'particles.star'}")
    raw = [r for r in table if "raw_mean" in r]
    if raw:
        mu, sd = np.median([r["raw_mean"] for r in raw]), np.median([r["raw_background_sd"] for r in raw])
        print(f"particle images normalised (background mean 0, SD 1); before: mean {mu:.4g}, background SD {sd:.4g} "
              f"(medians over micrographs) -> {out / 'stacks'}")
    if poses:
        ang = np.array([p["angle_deg"] for p in poses])
        with open(out / "poses.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(poses[0]))
            w.writeheader()
            w.writerows(poses)
        print(f"orientation vs reference ({len(ang)} paired particles, smallest angle over the symmetry operators): "
              f"median {np.median(ang):.2f} deg, 90th percentile {np.percentile(ang, 90):.2f} deg, "
              f"{100 * np.mean(ang < 2):.0f}% below 2 deg -> {out / 'poses.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
