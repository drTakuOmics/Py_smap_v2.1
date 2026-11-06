#!/usr/bin/env python
"""Control searches for the DeCo-LACE TRiC run (written next to the finished tiles).

  filled:     tile(s) with the strong-artifact zones (metal layer, edges) replaced by Gaussian noise
              of the remaining area's mean/SD; searched with the LSU (did the artifact hide the
              ribosomes?) and with TRiC (do the edge hits disappear?).
  phaserand:  TRiC on phase-randomised tiles (same power spectrum, no molecules): hits >= cutoff
              per tile from noise alone.

  cd $PYSMAP
  python examples/decolace_tric/make_controls.py --dest $WORK --filled 00520 --phaserand 00463 00453
Writes images_filled/, pars_controls/, jobs_controls.txt; results go to results_controls/<name>/.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np


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
    return out + [f"{k} {v}" for k, v in values.items() if k not in done]


def find(dirpath: Path, key: str) -> Path:
    hits = sorted(p for p in dirpath.glob("*.par") if key in p.stem)
    if len(hits) != 1:
        raise SystemExit(f"{key}: {len(hits)} matches in {dirpath}")
    return hits[0]


def main(argv=None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (smap)
    from smap import io as sio
    spec = importlib.util.spec_from_file_location("rev", Path(__file__).with_name("decolace_review.py"))
    rev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rev)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", required=True)
    ap.add_argument("--filled", nargs="*", default=[], help="tile number(s), e.g. 00520")
    ap.add_argument("--phaserand", nargs="*", default=[], help="tile number(s) for phase-randomised TRiC")
    ap.add_argument("--margin-A", type=float, default=60.0, help="margin around the artifact zones")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    d = Path(a.dest).expanduser().resolve()
    pc = d / "pars_controls"
    pc.mkdir(exist_ok=True)
    jobs = []

    def write(src: Path, name: str, extra: dict):
        lines = set_keys(src.read_text().splitlines(), dict(extra, outputDir=d / "results_controls" / name))
        par = pc / f"{name}.par"
        par.write_text("\n".join(lines) + "\n")
        jobs.append(str(par))

    for key in a.filled:
        src_l, src_t = find(d / "pars_LSU_control", key), find(d / "pars_TRiC", key)
        p, _ = sio.read_params_file(src_t)
        im = sio.mr(p["imageFile"]).astype(np.float32)
        pix = float(p["aPerPix"])
        art, b = rev.artifact_mask(im, pix, 5.0, a.margin_A)
        up = np.zeros(im.shape, bool)
        k = np.kron(art, np.ones((b, b), bool))
        up[:k.shape[0], :k.shape[1]] = k
        keep = ~up
        rng = np.random.default_rng(a.seed)
        out = im.copy()
        out[up] = rng.normal(im[keep].mean(), im[keep].std(), int(up.sum())).astype(np.float32)
        fn = d / "images_filled" / f"{Path(p['imageFile']).stem}_filled.mrc"
        fn.parent.mkdir(exist_ok=True)
        sio.mw(out, fn, pix)
        print(f"{key}: filled {100 * up.mean():.1f}% of the image (artifact zones + {a.margin_A:g} A) -> {fn.name}")
        write(src_l, f"LSU_{key}_filled", {"imageFile": fn})
        write(src_t, f"TRiC_{key}_filled", {"imageFile": fn})
    for key in a.phaserand:
        write(find(d / "pars_TRiC", key), f"TRiC_{key}_phaserand{a.seed}",
              {"decoy_image": "phase_random", "decoy_seed": a.seed})
    (d / "jobs_controls.txt").write_text("\n".join(jobs) + "\n")
    print(f"{len(jobs)} control searches -> {d / 'jobs_controls.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
