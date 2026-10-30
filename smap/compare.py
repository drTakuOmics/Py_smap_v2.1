"""Compare two SMAP output folders (e.g. MATLAB v2.1 vs this port, CPU vs GPU, batch 1 vs 8).

usage:  python -m smap.compare REFERENCE_OUTPUT_DIR TEST_OUTPUT_DIR [--json report.json]

Particles are paired one-to-one by search-grid position (list_final.txt columns 2-3), the
same pairing is used for particles.txt / particles.par (row j of each file belongs to row j
of list_final.txt). Reported per pair: final and refined score differences, positions,
defocus and the rotation angle between the two orientations. Search-stage maps
(search_mean/SD.mrc), survival curves (search_SH.txt) and highVals.txt are compared too.
Tolerances are declared on the command line and recorded in the report; exit status 1
means at least one check exceeded its tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import io as sio

DEFAULT_TOL = dict(score=0.05, grid_px=1.0, opt_px=0.5, angle_deg=1.0, defocus_nm=1.5,
                   map_rel=1e-3, sh_log10=0.05)


def _matrix(path: Path, skiprows: int = 0) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 0))
    lines = [x for x in path.read_text().splitlines()[skiprows:]
             if x.strip() and not x.lstrip().startswith("#")]
    return np.loadtxt(lines, ndmin=2) if lines else np.empty((0, 0))


def _quat_angle(qa, qb) -> np.ndarray:
    qa = qa / np.linalg.norm(qa, axis=-1, keepdims=True)
    qb = qb / np.linalg.norm(qb, axis=-1, keepdims=True)
    d = np.clip(np.abs(np.sum(qa * qb, axis=-1)), 0, 1)
    return 2 * np.degrees(np.arccos(d))


def _frealign_R(psi, theta, phi) -> np.ndarray:
    def rz(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

    def ry(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
    return rz(math.radians(phi)) @ ry(math.radians(theta)) @ rz(math.radians(psi))


def _pair(xy_a, xy_b, radius):
    if len(xy_a) == 0 or len(xy_b) == 0:
        return []
    d = np.linalg.norm(xy_a[:, None] - xy_b[None], axis=2)
    big = 1e6
    cost = np.where(d <= radius, d, big)
    ia, ib = linear_sum_assignment(cost)
    return [(int(i), int(j), float(d[i, j])) for i, j in zip(ia, ib) if d[i, j] <= radius]


def compare_outputs(ref: Path, test: Path, tol: dict | None = None) -> dict:
    tol = {**DEFAULT_TOL, **(tol or {})}
    ref, test = Path(ref), Path(test)
    rep: dict = dict(reference=str(ref.resolve()), test=str(test.resolve()), tolerances=tol, checks={})
    chk = rep["checks"]

    # ---- particles
    la, lb = _matrix(ref / "list_final.txt"), _matrix(test / "list_final.txt")
    pa, pb = _matrix(ref / "particles.txt"), _matrix(test / "particles.txt")
    ra, rb = _matrix(ref / "particles.par", 1), _matrix(test / "particles.par", 1)
    na, nb = len(la), len(lb)
    pairs = _pair(la[:, 1:3] if na else np.empty((0, 2)), lb[:, 1:3] if nb else np.empty((0, 2)),
                  tol["grid_px"])
    rows = []
    for i, j, dpx in pairs:
        r = dict(ref_index=i + 1, test_index=j + 1, grid_distance_px=dpx,
                 d_score_refined=float(lb[j, 0] - la[i, 0]),
                 d_defocus_nm=float(np.max(np.abs(lb[j, 3:5] - la[i, 3:5]))),
                 angle_refined_deg=float(_quat_angle(la[i, 6:10], lb[j, 6:10])))
        if len(pa) > i and len(pb) > j:
            r["d_score_final"] = float(pb[j, 1] - pa[i, 1])
            r["score_final_ref"], r["score_final_test"] = float(pa[i, 1]), float(pb[j, 1])
        if len(ra) > i and len(rb) > j:
            Ra = _frealign_R(*ra[i, 1:4])
            Rb = _frealign_R(*rb[j, 1:4])
            c = np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1)
            r["angle_final_deg"] = float(np.degrees(np.arccos(c)))
        rows.append(r)
    paired_a = {r["ref_index"] for r in rows}
    paired_b = {r["test_index"] for r in rows}
    rep["particles"] = dict(n_reference=na, n_test=nb, n_paired=len(rows), pairs=rows,
                            unpaired_reference=[k for k in range(1, na + 1) if k not in paired_a],
                            unpaired_test=[k for k in range(1, nb + 1) if k not in paired_b])

    def worst(key):
        v = [abs(r[key]) for r in rows if key in r]
        return max(v) if v else None

    chk["same_particle_count"] = dict(value=[na, nb], ok=na == nb and len(rows) == na)
    for key, tk in [("d_score_final", "score"), ("d_score_refined", "score"),
                    ("angle_refined_deg", "angle_deg"), ("angle_final_deg", "angle_deg"),
                    ("d_defocus_nm", "defocus_nm")]:
        w = worst(key)
        chk[key] = dict(max_abs=w, tol=tol[tk], ok=(w is None or w <= tol[tk]))
    for cut in (7.0, 8.0):
        ca = int(np.sum(pa[:, 1] >= cut)) if len(pa) else 0
        cb = int(np.sum(pb[:, 1] >= cut)) if len(pb) else 0
        chk[f"count_final_ge_{cut:g}"] = dict(value=[ca, cb], ok=ca == cb)

    # ---- search-stage maps
    for name in ("search_mean.mrc", "search_SD.mrc"):
        if (ref / name).exists() and (test / name).exists():
            A, B = sio.mr(ref / name).astype(np.float64), sio.mr(test / name).astype(np.float64)
            if A.shape != B.shape:
                chk[name] = dict(value=[list(A.shape), list(B.shape)], ok=False)
                continue
            scale = float(np.std(A)) or 1.0
            m = float(np.max(np.abs(A - B)))
            chk[name] = dict(max_abs=m, relative=m / scale, tol=tol["map_rel"], ok=m / scale <= tol["map_rel"])

    sa, sb = _matrix(ref / "search_SH.txt"), _matrix(test / "search_SH.txt")
    if len(sa) and len(sb):
        lo, hi = max(sa[0, 0], sb[0, 0]), min(sa[-1, 0], sb[-1, 0])
        xs = np.linspace(lo, hi, 200)
        ya = np.interp(xs, sa[:, 0], sa[:, 2])
        yb = np.interp(xs, sb[:, 0], sb[:, 2])
        ok = (ya >= 10) & (yb >= 10)
        d = float(np.max(np.abs(np.log10(ya[ok]) - np.log10(yb[ok])))) if ok.any() else None
        chk["search_SH_tail"] = dict(max_abs_log10=d, tol=tol["sh_log10"], ok=d is None or d <= tol["sh_log10"])

    ha, hb = _matrix(ref / "highVals.txt"), _matrix(test / "highVals.txt")
    if len(ha) or len(hb):
        ka = {(int(r[0]), int(r[2]), int(r[3]), int(r[4])): r[1] for r in ha}
        kb = {(int(r[0]), int(r[2]), int(r[3]), int(r[4])): r[1] for r in hb}
        common = set(ka) & set(kb)
        d = max((abs(ka[k] - kb[k]) for k in common), default=None)
        chk["highVals"] = dict(n=[len(ka), len(kb)], n_common=len(common), max_abs_score=d,
                               ok=(len(ka) == len(kb) == len(common)) and (d is None or d <= tol["score"]),
                               note="rotation indices of repeated grid orientations can differ")
    rep["all_ok"] = all(c["ok"] for c in chk.values())
    return rep


def format_report(rep: dict) -> str:
    out = [f"reference: {rep['reference']}", f"test:      {rep['test']}", ""]
    out.append(f"{'check':28s} {'result':>32s}  {'tol':>8s}  ok")
    for k, c in rep["checks"].items():
        val = c.get("max_abs", c.get("max_abs_log10", c.get("value", c.get("n"))))
        if isinstance(val, float):
            val = f"{val:.4g}"
        out.append(f"{k:28s} {str(val):>32s}  {str(c.get('tol', '')):>8s}  {'yes' if c['ok'] else 'NO'}")
    p = rep["particles"]
    out.append("")
    out.append(f"particles: {p['n_reference']} reference, {p['n_test']} test, {p['n_paired']} paired; "
               f"unpaired ref {p['unpaired_reference']}, test {p['unpaired_test']}")
    if p["pairs"]:
        out.append(f"{'ref':>4s} {'test':>4s} {'final_ref':>9s} {'final_test':>10s} {'d_final':>8s} "
                   f"{'ang_deg':>8s} {'d_df_nm':>8s}")
        for r in p["pairs"]:
            out.append(f"{r['ref_index']:4d} {r['test_index']:4d} {r.get('score_final_ref', float('nan')):9.3f} "
                       f"{r.get('score_final_test', float('nan')):10.3f} {r.get('d_score_final', float('nan')):8.3f} "
                       f"{r.get('angle_final_deg', float('nan')):8.3f} {r['d_defocus_nm']:8.2f}")
    out.append("")
    out.append("ALL CHECKS WITHIN TOLERANCE" if rep["all_ok"] else "SOME CHECKS EXCEED TOLERANCE")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("test")
    ap.add_argument("--json", default=None, help="write the full report to this file")
    for k, v in DEFAULT_TOL.items():
        ap.add_argument(f"--tol-{k.replace('_', '-')}", type=float, default=v, dest=f"tol_{k}")
    a = ap.parse_args(argv)
    tol = {k: getattr(a, f"tol_{k}") for k in DEFAULT_TOL}
    rep = compare_outputs(Path(a.reference), Path(a.test), tol)
    print(format_report(rep))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
    return 0 if rep["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
