"""Quality control of detected particles (new; not part of v2.1).

Inspired by the particle filters of Zhang et al., eLife 2026 (2DTM of a 43 kDa kinase) and
written independently for SMAP outputs. Each candidate in ``particles.csv`` is described
by features read at its search-grid pixel, and is flagged when a feature does not fit.

Per-particle features (search grid, best defocus plane of that pixel):

* ``cc_mean``, ``cc_sd``  mean and SD of the cross-correlation over all orientations
  (``search_mean.mrc``, ``search_SD.mrc``).
* ``z_search``, ``neg_log_p``  flat-fielded score and -ln p of that pixel.
* ``prominence``  z at the peak minus the median z in an annulus around it.
* ``anisotropy``  ratio of the principal curvatures of the z-map at the peak (Hessian of a
  smoothed map). Particles give round peaks (ratio near 1); edges and filaments give ridges.

How the mean/SD test works. Fixed cut-offs such as "mean < 0" or "SD > 1.1" depend on how
the correlation is normalised, and a strong, true particle also raises the mean and SD over
orientations (its projections correlate with many orientations at low resolution). Here
both effects are modelled. With the background level removed, a target of peak signal x
(raw MIP minus background mean) adds about alpha*x to the mean and gamma*x^2 to the
variance over orientations, with alpha and gamma set by the template's rotational
self-similarity. A low-resolution artefact (ice edge, carbon, contamination) correlates
with many orientations at once and gives much larger ratios. alpha and gamma are estimated
per image as the median ratios over its candidates, and a candidate is flagged when its
ratio lies more than k (= 5) times its uncertainty from that median; the uncertainty is the
population's robust SD combined with the candidate's own noise (background spread of the
mean or variance map divided by its signal), so weak candidates are not flagged for noise. This needs at least
``min_population`` (6) candidates; with fewer, only an anti-correlated mean (below the
background median by k robust SDs) is flagged.

Image-level filters (CTF fit score, sample thickness) are applied when a table with those
columns is given.

Output: ``particles_qc.csv`` (one row per particle: features, flags, ``qc_pass``) and
``particles_qc.json`` (fitted ratios, bounds and counts).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import io as sio

MAD_TO_SD = 1.4826


def robust_bounds(values: np.ndarray, k: float) -> tuple[float, float, float, float]:
    """(median, robust SD, lower, upper) with robust SD = 1.4826 * MAD."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    med = float(np.median(v))
    rsd = float(MAD_TO_SD * np.median(np.abs(v - med)))
    return med, rsd, med - k * rsd, med + k * rsd


def _plane_values(vol: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """vol[r, c, plane[r, c]] for a 3D (rows, cols, planes) volume; 2D volumes pass through."""
    if vol.ndim == 2:
        return vol
    p = np.clip(plane.astype(int), 0, vol.shape[2] - 1)
    return np.take_along_axis(vol, p[:, :, None], 2)[:, :, 0]


def prominence(z: np.ndarray, r: int, c: int, r_in: float, r_out: float) -> float:
    rr, cc = np.ogrid[:z.shape[0], :z.shape[1]]
    d = np.hypot(rr - r, cc - c)
    ring = (d >= r_in) & (d <= r_out) & (z != 0)
    return float(z[r, c] - np.median(z[ring])) if ring.any() else float("nan")


def anisotropy(z_smooth: np.ndarray, r: int, c: int) -> float:
    """|lambda_max| / |lambda_min| of the 2x2 Hessian at (r, c) (finite differences)."""
    if not (1 <= r < z_smooth.shape[0] - 1 and 1 <= c < z_smooth.shape[1] - 1):
        return float("nan")
    f = z_smooth
    frr = f[r + 1, c] - 2 * f[r, c] + f[r - 1, c]
    fcc = f[r, c + 1] - 2 * f[r, c] + f[r, c - 1]
    frc = (f[r + 1, c + 1] - f[r + 1, c - 1] - f[r - 1, c + 1] + f[r - 1, c - 1]) / 4
    ev = np.abs(np.linalg.eigvalsh(np.array([[frr, frc], [frc, fcc]])))
    return float(ev.max() / ev.min()) if ev.min() > 0 else float("inf")


def load_maps(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    need = ["search_mip_scaled.mrc", "search_mean.mrc", "search_SD.mrc"]
    missing = [n for n in need if not (out_dir / n).exists()]
    if missing:
        raise FileNotFoundError(f"{out_dir}: missing {missing} (run the search with write_mip 1)")
    z = sio.mr(out_dir / "search_mip_scaled.mrc").astype(np.float64)
    planes = (sio.mr(out_dir / "search_mip_dfInds.mrc").astype(int) - 1
              if (out_dir / "search_mip_dfInds.mrc").exists() else np.zeros(z.shape, int))
    mean = _plane_values(sio.mr(out_dir / "search_mean.mrc").astype(np.float64), planes)
    sd = _plane_values(sio.mr(out_dir / "search_SD.mrc").astype(np.float64), planes)
    pv = sio.mr(out_dir / "search_pval.mrc").astype(np.float64) if (out_dir / "search_pval.mrc").exists() else None
    return dict(z=z, mean=mean, sd=sd, pv=pv, valid=z != 0)


def read_image_table(path) -> dict:
    """image -> {ctf_score, thickness_nm} from a CSV with columns image[,ctf_score][,thickness_nm]."""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[r["image"]] = {k: float(r[k]) for k in ("ctf_score", "thickness_nm") if r.get(k, "") != ""}
    return out


def _ratio_flags(x, y, k: float, noise_sd: float):
    """Median ratio y/x over candidates with x > 0 and outlier flags.

    A candidate's ratio is compared with the median using the population's robust SD plus
    the noise of its own ratio (noise_sd / x): weak candidates have noisier ratios.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & np.isfinite(y)
    ratio = np.full(x.shape, np.nan)
    ratio[ok] = y[ok] / x[ok]
    if not ok.any():
        return ratio, float("nan"), float("nan"), ok & False, ok & False
    med = float(np.nanmedian(ratio))
    rsd = float(MAD_TO_SD * np.nanmedian(np.abs(ratio - med)))
    scale = np.full(x.shape, np.inf)
    scale[ok] = np.sqrt(rsd ** 2 + (noise_sd / x[ok]) ** 2)
    dev = (ratio - med) / scale
    return ratio, med, rsd, ok & (dev > k), ok & (dev < -k)


def run_qc(out_dir, k: float = 5.0, particle_radius_px: float = 10.0, image_info: dict | None = None,
           ctf_window=None, thickness_window=None, max_anisotropy: float | None = None,
           min_population: int = 6) -> dict:
    out_dir = Path(out_dir)
    m = load_maps(out_dir)
    valid = m["valid"]
    b_mean = robust_bounds(m["mean"][valid], k)
    b_sd = robust_bounds(m["sd"][valid], k)
    z_s = ndimage.gaussian_filter(m["z"], sigma=max(1.0, particle_radius_px / 3))

    img_flag = []
    if image_info is not None:
        if ctf_window and "ctf_score" in image_info and not ctf_window[0] <= image_info["ctf_score"] <= ctf_window[1]:
            img_flag.append("ctf_score")
        if thickness_window and "thickness_nm" in image_info and \
                not thickness_window[0] <= image_info["thickness_nm"] <= thickness_window[1]:
            img_flag.append("thickness")

    rows = []
    with open(out_dir / "particles.csv") as fh:
        for p in csv.DictReader(fh):
            r, c = int(round(float(p["row_search"]))) - 1, int(round(float(p["col_search"]))) - 1
            inside = 0 <= r < valid.shape[0] and 0 <= c < valid.shape[1]
            nan = math.nan
            f = dict(num_id=int(p["num_id"]), score_final=float(p["score_final"]),
                     cc_mean=float(m["mean"][r, c]) if inside else nan,
                     cc_sd=float(m["sd"][r, c]) if inside else nan,
                     z_search=float(m["z"][r, c]) if inside else nan,
                     neg_log_p=float(m["pv"][r, c]) if inside and m["pv"] is not None else nan,
                     prominence=prominence(m["z"], r, c, particle_radius_px, 2 * particle_radius_px) if inside else nan,
                     anisotropy=anisotropy(z_s, r, c) if inside else nan)
            f["signal"] = f["cc_mean"] + f["z_search"] * f["cc_sd"] - b_mean[0]     # raw MIP - background mean
            f["_flags"] = list(img_flag) + ([] if inside else ["outside_grid"])
            rows.append(f)

    x = np.array([r["signal"] for r in rows]) if rows else np.zeros(0)
    model = dict(population=len(rows), used=len(rows) >= min_population)
    if model["used"]:
        y_mean = np.array([r["cc_mean"] - b_mean[0] for r in rows])
        y_var = np.array([r["cc_sd"] ** 2 - b_sd[0] ** 2 for r in rows])
        rm, am, sm, hi_m, lo_m = _ratio_flags(x, y_mean, k, noise_sd=b_mean[1])
        rv, gv, sv, hi_v, _ = _ratio_flags(x ** 2, y_var, k, noise_sd=2 * b_sd[0] * b_sd[1])
        model.update(alpha=am, alpha_robust_sd=sm, gamma=gv, gamma_robust_sd=sv)
        for i, row in enumerate(rows):
            row["mean_ratio"], row["var_ratio"] = float(rm[i]), float(rv[i])
            if hi_m[i]:
                row["_flags"].append("cc_mean_high")
            if lo_m[i]:
                row["_flags"].append("cc_mean_low")
            if hi_v[i]:
                row["_flags"].append("cc_sd_high")
    for row in rows:
        row.setdefault("mean_ratio", math.nan)
        row.setdefault("var_ratio", math.nan)
        if not model["used"] and row["cc_mean"] < b_mean[2]:
            row["_flags"].append("cc_mean_low")
        if max_anisotropy is not None and row["anisotropy"] > max_anisotropy:
            row["_flags"].append("ridge")
        row["flags"] = ";".join(row.pop("_flags"))
        row["qc_pass"] = int(not row["flags"])

    summary = dict(out_dir=str(out_dir), k=k, n_particles=len(rows), n_pass=sum(r["qc_pass"] for r in rows),
                   signal_model=model,
                   background=dict(mean_median=b_mean[0], mean_robust_sd=b_mean[1],
                                   sd_median=b_sd[0], sd_robust_sd=b_sd[1]),
                   image_flags=img_flag, max_anisotropy=max_anisotropy,
                   note=("mean/SD flags: ratio of excess mean (variance) to signal (signal^2) more than k robust "
                         "SDs from the image's median ratio; needs >= min_population candidates"))
    return dict(rows=rows, summary=summary)


def write_qc(res: dict, out_dir) -> None:
    out_dir = Path(out_dir)
    rows = res["rows"]
    cols = ["num_id", "score_final", "z_search", "neg_log_p", "cc_mean", "cc_sd", "signal", "mean_ratio",
            "var_ratio", "prominence", "anisotropy", "flags", "qc_pass"]
    with open(out_dir / "particles_qc.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    (out_dir / "particles_qc.json").write_text(json.dumps(res["summary"], indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Flag outlier particles of a finished search (writes particles_qc.csv)")
    ap.add_argument("output_dir")
    ap.add_argument("--k", type=float, default=5.0, help="robust-SD multiplier for the per-image bounds")
    ap.add_argument("--radius", type=float, default=10.0, help="particle radius on the search grid (pixels)")
    ap.add_argument("--images", help="CSV with columns image, ctf_score, thickness_nm")
    ap.add_argument("--image-name", help="row of --images that this output belongs to")
    ap.add_argument("--ctf-window", type=float, nargs=2)
    ap.add_argument("--thickness-window", type=float, nargs=2, help="nm")
    ap.add_argument("--max-anisotropy", type=float, help="flag peaks that look like ridges")
    ap.add_argument("--min-population", type=int, default=6,
                    help="candidates needed per image to fit the signal model for the mean/SD flags")
    a = ap.parse_args(argv)
    info = None
    if a.images:
        table = read_image_table(a.images)
        info = table.get(a.image_name or Path(a.output_dir).name)
    res = run_qc(a.output_dir, a.k, a.radius, info, a.ctf_window, a.thickness_window, a.max_anisotropy,
                 a.min_population)
    write_qc(res, a.output_dir)
    s = res["summary"]
    print(f"{s['n_pass']}/{s['n_particles']} particles pass QC -> {Path(a.output_dir) / 'particles_qc.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
