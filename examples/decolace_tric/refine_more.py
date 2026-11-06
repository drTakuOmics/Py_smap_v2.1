#!/usr/bin/env python
"""Refine more candidates of a finished 3.8 deg search, without searching again.

The search keeps every flat-fielded value above arbThr (6.0) in search_vals.mrc / search_qInds.mrc,
but refines only clusters above optThr (7.0). Real particles that fall between two grid
orientations score 6-7 in the search and are never refined (28 of the authors' 84 ribosomes on
tile 463). This re-runs clustering, refinement (0.5 deg local grid, defocus scans) and optimisation
from those saved maps with a lower optThr; the search itself is not repeated.

Threshold: the lowest of 6.0 / 6.25 / 6.5 / 6.75 that gives at most --max-candidates clusters
(or --opt-thr). The final cutoff (search_status.json) is unchanged, so hits are judged exactly as
before; only more candidates get the chance to be refined.

  cd $PYSMAP
  python examples/decolace_tric/refine_more.py --dest $WORK --target LSU_control --tiles 00463 --count-only
  python examples/decolace_tric/refine_more.py --dest $WORK --target LSU_control --tiles 00463
Writes results_<target>_opt<thr>/<tile>/ (new particles.csv; the search maps are linked, not copied).
CPU by default (the GPU stays with the queue); --device cuda when the GPU is free.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

LINK = ["search_mip_scaled.mrc", "search_mip.mrc", "search_mip_qInds.mrc", "search_mip_dfInds.mrc",
        "search_pval.mrc", "search_mean.mrc", "search_SD.mrc", "search_vals.mrc", "search_qInds.mrc", "cutoffs.json"]


def load_maps(src: Path):
    from smap import io as sio
    need = ["search_vals.mrc", "search_qInds.mrc", "search_mean.mrc", "search_SD.mrc", "search_status.json"]
    miss = [n for n in need if not (src / n).exists()]
    if miss:
        raise SystemExit(f"{src}: missing {miss}")
    ff = sio.mr(src / "search_vals.mrc").astype(np.float64)
    fi = sio.mr(src / "search_qInds.mrc").astype(np.float64)
    if ff.ndim == 2:
        ff, fi = ff[:, :, None], fi[:, :, None]
    return ff, fi, json.loads((src / "search_status.json").read_text())


def count_clusters(ff: np.ndarray, thr: float) -> int:
    """Number of clusters the selection step would refine (8-connected regions per defocus plane)."""
    from scipy import ndimage
    return int(sum(ndimage.label(ff[:, :, i] > thr, structure=np.ones((3, 3)))[1] for i in range(ff.shape[2])))


def choose(ff, st0, thr, max_cand, log=print):
    arb = float(st0.get("raw_preselection_threshold", 6.0))
    counts = {t: count_clusters(ff, t) for t in (6.0, 6.25, 6.5, 6.75, 7.0)}
    log("clusters per optThr (before merging planes): " + ", ".join(f"{t:g}: {n}" for t, n in counts.items()))
    if thr is None:
        ok = [t for t, n in counts.items() if n <= max_cand]
        thr = min(ok) if ok else 7.0
    return max(thr, arb), counts


def rerefine(par: Path, src: Path, out: Path, thr: float, device: str, log=print) -> int:
    from smap import io as sio
    from smap.refine import optimize_particles, refine_particles, select_particles, write_final
    from smap.search_global import prepare
    ff, fi, st0 = load_maps(src)
    out.mkdir(parents=True, exist_ok=True)
    st = prepare(par, device=device, write_files=True, output_dir=out)
    if tuple(st.Npix_im) != ff.shape[:2] or st.defocus.shape[0] != ff.shape[2]:
        raise SystemExit(f"{src}: saved maps {ff.shape} do not match this .par ({st.Npix_im}, {st.defocus.shape[0]} planes)")
    if st.R_full.shape[0] < int(fi.max()):
        raise SystemExit(f"{src}: rotation index {int(fi.max())} > grid size {st.R_full.shape[0]}; different angle_inc?")
    for n in LINK:                                   # maps for the cutoff / review tools
        s, d = src / n, out / n
        if s.exists() and not d.exists():
            try:
                d.symlink_to(s.resolve())
            except OSError:
                shutil.copy(s, d)
    st_new = dict(st0, refine_more=dict(source=str(src), optThr=thr, date=time.strftime("%Y-%m-%d %H:%M")))
    (out / "search_status.json").write_text(json.dumps(st_new, indent=1))
    st.params["optThr"] = thr
    comb = dict(ff=ff, ff_inds=fi, mean=sio.mr(src / "search_mean.mrc").astype(np.float64),
                sd=sio.mr(src / "search_SD.mrc").astype(np.float64))
    t0 = time.time()
    jobs = select_particles(st, comb)
    parts = refine_particles(st, jobs, comb["mean"], comb["sd"], device=device, log=log) if jobs else []
    opt = optimize_particles(st, parts, device=device, log=log) if parts else []
    write_final(st, opt)
    (out / ".done").touch()
    cut = float(st0.get("gaussian_threshold") or 7.42)
    n_cut = sum(o["peak_opt"] >= cut for o in opt)
    log(f"{len(opt)} refined in {(time.time() - t0) / 60:.1f} min; {n_cut} >= cutoff {cut:.2f}")
    return n_cut


def main(argv=None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (smap)
    try:
        sys.stdout.reconfigure(line_buffering=True)      # progress visible in a log file without -u
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", required=True)
    ap.add_argument("--target", required=True,
                    help="LSU_control, LSU_mouse, TRiC, TRiC_decoy, TRiC_open, controls (pars_<target>/)")
    ap.add_argument("--tiles", nargs="+", required=True, help="tile numbers, or: all (every finished search)")
    ap.add_argument("--opt-thr", type=float, default=None)
    ap.add_argument("--max-candidates", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--count-only", action="store_true")
    a = ap.parse_args(argv)
    d = Path(a.dest).expanduser().resolve()
    tiles = a.tiles
    if tiles == ["all"]:
        import re
        tiles = sorted({m.group(1) for p in (d / f"results_{a.target}").glob("*/.done")
                        if (m := re.search(r"_(\d{5})_", p.parent.name))})
        print(f"{a.target}: {len(tiles)} finished searches: {' '.join(tiles)}")
    for k in tiles:
        pars = sorted((d / f"pars_{a.target}").glob(f"*_{k}_*.par"))
        if not pars:
            print(f"{k}: no .par in pars_{a.target}")
            continue
        par = pars[0]
        src = d / f"results_{a.target}" / par.stem
        if not (src / ".done").exists() and not (src / "particles.csv").exists():
            print(f"{k}: search not finished ({src})")
            continue
        print(f"== {a.target} {k}", flush=True)
        ff, _, st0 = load_maps(src)
        thr, _ = choose(ff, st0, a.opt_thr, a.max_candidates)
        print(f"optThr {thr:g}", flush=True)
        if a.count_only:
            continue
        out = d / f"results_{a.target}_opt{thr:g}" / par.stem
        if (out / ".done").exists():
            print(f"{k}: already refined ({out})")
            continue
        rerefine(par, src, out, thr, a.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
