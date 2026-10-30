"""E4: accuracy of the final (stage C) optimisation on synthetic ground truth.

Particles are planted at OFF-grid orientations (grid rotation x random rotation of up to
half the search step), sub-pixel positions and a defocus offset from the search plane.
Search, clustering and grid refinement (stage B) run once per image; stage C then runs in
three modes from the same stage-B result:

  v21         v2.1 behaviour (optimize_angle ignores its rotation parameters)
  fix_angle   fix_optimize_angle 1 (v2.1 optimiser, rotation actually used)
  continuous  refine_continuous 1 (rotation vector + shift + defocus, scaled simplex)

Reported per mode: angular error (deg) of q_opt vs the planted rotation, position error
(px) of x/y_orig0_est vs the planted centre, defocus error (nm) and final score.

usage: python experiments/e4_refinement_accuracy.py [--images 3] [--snr 14] [--out e4]
CPU, ~1-3 min per image.
"""

import argparse
import csv
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tests")
from smap import io as sio  # noqa: E402
from smap.ctf import ctf  # noqa: E402
from smap.refine import optimize_particles, refine_particles, select_particles  # noqa: E402
from smap.rot import rodrigues  # noqa: E402
from smap.search_global import _templates, combine, prepare, run_job  # noqa: E402
from smap.utils import normalize_rm  # noqa: E402
from synthetic import N, volume, write_par  # noqa: E402

IM = 240
CENTRES = [(62, 63), (63, 178), (177, 61), (178, 177)]
MODES = {"v21": {}, "fix_angle": {"fix_optimize_angle": 1}, "continuous": {"refine_continuous": 1}}


def template_for(st, R):
    Vr = torch.as_tensor(np.ascontiguousarray(st.V_F.real))
    Vi = torch.as_tensor(np.ascontiguousarray(st.V_F.imag))
    a = torch.arange(1, N + 1, dtype=torch.float64)
    rr, cc = torch.meshgrid(a, a, indexing="ij")
    cp = N // 2 + 1
    return _templates(Vr, Vi, torch.as_tensor(R[None]), cc - cp, rr - cp, st.V_F.shape[0] // 2 + 1)[0].numpy()


def make_image(d: Path, seed: int, snr: float, ddf_nm: float):
    rng = np.random.default_rng(seed)
    sio.mw(np.transpose(volume(np.random.default_rng(0)) * 0.002, (1, 2, 0)), d / "vol.mrc", 2.0)
    sio.mw(rng.standard_normal((IM, IM)).astype(np.float32), d / "dummy.mrc", 2.0)
    st0 = prepare(write_par(d, d / "dummy.mrc"), write_files=False)
    inc = 20.0
    df_true = st0.defocus[0].copy()
    df_true[:2] += ddf_nm
    Ci = np.imag(np.fft.ifftshift(ctf(df_true[None], [IM, IM], st0.params)[:, :, 0]))
    gr = np.fft.fftfreq(IM)[:, None]
    gc = np.fft.fftfreq(IM)[None, :]
    sig, truth = np.zeros((IM, IM)), []
    for (r0, c0) in CENTRES:
        k = int(rng.integers(1, st0.R_full.shape[0] + 1))
        v = rng.normal(size=3)
        v *= math.radians(rng.uniform(0.2, 0.5) * inc) / np.linalg.norm(v)
        R = normalize_rm(rodrigues(v) @ st0.R_full[k - 1])
        dr, dc = rng.uniform(-0.5, 0.5, 2)
        one = np.zeros((IM, IM))
        h = N // 2
        one[r0 - h:r0 - h + N, c0 - h:c0 - h + N] = template_for(st0, R)
        one = np.real(np.fft.ifft2(np.fft.fft2(one) * Ci * np.exp(-2j * np.pi * (gr * dr + gc * dc))))
        sig += one * snr / np.linalg.norm(one)
        truth.append(dict(R=R, x=c0 + dc, y=r0 + dr, df=df_true[0], k=k,
                          off_grid_deg=math.degrees(np.linalg.norm(v))))
    sio.mw((sig + rng.standard_normal((IM, IM))).astype(np.float32), d / "img.mrc", 2.0)
    return d / "img.mrc", truth


def angle_deg(Ra, Rb):
    return math.degrees(math.acos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1)))


def run(n_images, snr, ddf, out):
    rows = []
    for seed in range(n_images):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            img, truth = make_image(d, seed + 10, snr, ddf)
            st = prepare(write_par(d, img, extra="optThr 5.0\ninc_degrees 2\nrefine_local_peak 20\n"))
            for j in (1, 2):
                run_job(st, j, device="cpu", batch_size=16)
            res = combine(st)
            parts = refine_particles(st, select_particles(st, res), res["mean"], res["sd"], device="cpu",
                                     log=lambda *a: None)
            base = dict(st.params)
            for mode, opts in MODES.items():
                st.params.clear()
                st.params.update({**base, "fix_optimize_angle": 0, "refine_continuous": 0, **opts})
                t0 = time.time()
                opt = optimize_particles(st, [dict(p) for p in parts], device="cpu", log=lambda *a: None)
                dt = (time.time() - t0) / max(1, len(opt))
                for o in opt:
                    x, y = o["xy_orig0_est"]
                    dist = [math.hypot(x - t["x"], y - t["y"]) for t in truth]
                    i = int(np.argmin(dist))
                    if dist[i] > 5:
                        continue
                    t = truth[i]
                    rows.append(dict(image=seed, particle=i, mode=mode, off_grid_deg=t["off_grid_deg"],
                                     angle_err_deg=angle_deg(o["q_opt"], t["R"]),
                                     angle_err_stageB_deg=angle_deg(o["q"], t["R"]), pos_err_px=dist[i],
                                     df_err_nm=float(o["df"][0] - t["df"]), score_final=o["peak_opt"],
                                     score_refined=o["peak"], sec_per_particle=dt))
            print(f"image {seed}: {len(parts)} stage-B particles", flush=True)
    out = Path(out)
    with open(f"{out}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = {}
    for mode in MODES:
        r = [x for x in rows if x["mode"] == mode]
        summ[mode] = {k: dict(median=float(np.median([x[k] for x in r])),
                              mean=float(np.mean([x[k] for x in r])))
                      for k in ("angle_err_deg", "angle_err_stageB_deg", "pos_err_px", "score_final",
                                "sec_per_particle")}
        summ[mode]["abs_df_err_nm"] = dict(median=float(np.median([abs(x["df_err_nm"]) for x in r])))
        summ[mode]["n"] = len(r)
    stage_b = [x for x in rows if x["mode"] == "v21"]
    summ["setup"] = dict(images=n_images, snr=snr, defocus_offset_nm=ddf, grid_deg=20, refine_inc_deg=2,
                         off_grid_deg_median=float(np.median([x["off_grid_deg"] for x in stage_b])))
    Path(f"{out}.json").write_text(json.dumps(summ, indent=1))
    for mode in MODES:
        s = summ[mode]
        print(f"{mode:11s} n={s['n']:2d}  angle err {s['angle_err_deg']['median']:.2f} deg "
              f"(stage B {s['angle_err_stageB_deg']['median']:.2f})  "
              f"pos err {s['pos_err_px']['median']:.2f} px  |df err| {s['abs_df_err_nm']['median']:.1f} nm  "
              f"score {s['score_final']['median']:.2f}  {s['sec_per_particle']['median']:.1f} s/particle")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=3)
    ap.add_argument("--snr", type=float, default=14.0)
    ap.add_argument("--defocus-offset", type=float, default=12.0, help="nm, planted minus search plane")
    ap.add_argument("--out", default="e4_refinement")
    a = ap.parse_args()
    run(a.images, a.snr, a.defocus_offset, a.out)
