"""Threshold statistics for the flat-fielded search scores (new; not part of v2.1).

v2.1 sets thr = sqrt(2)*erfcinv(2/N), N = orientations x pixels x defocus planes, so that
N*Q(thr) = 1 expected *evaluation* above threshold under independent Gaussian noise.
Correlations between neighbouring pixels/orientations/defoci do not change that expected
evaluation count (each evaluation is still marginally Gaussian), so fitting the
survival curve of evaluations (search_SH.txt) cannot measure them. What correlation
does change is how evaluations group into separate *detections*: exceedances come in
clusters, so the expected number of false *particles* is lower than N*Q(thr).

This module therefore works on clusters:

* ``exact_survival``: count(score > x), strict (ties excluded);
* ``completeness_threshold``: below this x the flat-fielded tail is censored, because
  v2.1 keeps only raw CC > arbThr (an evaluation with normalised score x is retained
  only where mean + SD*x > arbThr);
* ``cluster_survival``: number of 8-connected regions of the per-pixel maximum map
  (search_vals.mrc) above x, summed over defocus planes;
* ``fit_neff``: c(x) ~= N_eff * Q(x) on the complete, well-populated range, giving the
  threshold for one expected false *cluster*.

Fit N_eff on particle-free control searches (e.g. a different protein of similar mass, or a particle-free
area). On real images, true particles inflate c(x); the value written by combine() is
informational only and is never used to select particles.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import erfc, erfcinv


def gaussian_tail(x):
    return erfc(np.asarray(x, dtype=np.float64) / math.sqrt(2)) / 2


def threshold_for(n_trials: float, n_false: float = 1.0) -> float:
    return float(math.sqrt(2) * erfcinv(2.0 * n_false / n_trials))


def exact_survival(scores, thresholds) -> np.ndarray:
    """count(scores > t) for each threshold t (strict inequality, ties excluded)."""
    s = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    return s.size - np.searchsorted(s, np.asarray(thresholds, dtype=np.float64), side="right")


def completeness_threshold(mean_im, sd_im, arb_thr: float, region) -> float:
    """Smallest x such that every kept pixel retains all evaluations with score > x."""
    m = np.asarray(mean_im)[region]
    s = np.asarray(sd_im)[region]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (arb_thr - m) / s
    x = x[np.isfinite(x)]
    return float(x.max()) if x.size else float("nan")


def cluster_survival(ff, xs) -> np.ndarray:
    """Number of 8-connected regions with ff > x (summed over planes) for each x."""
    from scipy import ndimage
    f = np.asarray(ff, dtype=np.float64)
    if f.ndim == 2:
        f = f[:, :, None]
    st = np.ones((3, 3))
    out = []
    for x in np.asarray(xs, dtype=np.float64):
        out.append(sum(ndimage.label(f[:, :, k] > x, structure=st)[1] for k in range(f.shape[2])))
    return np.array(out, dtype=np.int64)


def fit_neff(xs, counts, min_count: int = 10):
    """Fit log c(x) = log N_eff + log Q(x) over points with c >= min_count."""
    xs = np.asarray(xs, dtype=np.float64)
    c = np.asarray(counts, dtype=np.float64)
    ok = c >= min_count
    if ok.sum() < 3:
        return dict(N_eff=None, reason=f"fewer than 3 points with >= {min_count} clusters")
    lr = np.log(c[ok]) - np.log(gaussian_tail(xs[ok]))
    log_neff = float(np.mean(lr))
    return dict(N_eff=float(math.exp(log_neff)), x_min=float(xs[ok].min()),
                x_max=float(xs[ok].max()), n_points=int(ok.sum()),
                rms_log_residual=float(np.sqrt(np.mean((lr - log_neff) ** 2))))


def calibrate_threshold(ff, mean_im, sd_im, arb_thr, region2d, n_trials, n_points=30) -> dict:
    """Summary written by combine() to search_status.json (informational only)."""
    reg = np.broadcast_to(np.asarray(region2d)[:, :, None], np.asarray(mean_im).shape)
    x_c = completeness_threshold(mean_im, sd_im, arb_thr, reg)
    out = dict(x_complete=x_c, n_trials_v21=float(n_trials), threshold_v21=threshold_for(n_trials))
    f = np.asarray(ff)
    vals = f[f != 0]
    if np.isfinite(x_c) and vals.size and vals.max() > x_c:
        xs = np.linspace(x_c, float(vals.max()), n_points)
        c = cluster_survival(f, xs)
        out.update(cluster_x=xs.tolist(), cluster_counts=c.tolist(), fit=fit_neff(xs, c))
        if out["fit"].get("N_eff"):
            out["N_eff_over_N"] = out["fit"]["N_eff"] / n_trials
            out["threshold_N_eff"] = threshold_for(out["fit"]["N_eff"])
    out["note"] = ("cluster-based estimate; meaningful on particle-free control searches only; "
                   "not used for particle selection")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Print threshold calibration of a finished search")
    ap.add_argument("output_dir")
    a = ap.parse_args(argv)
    st = json.loads((Path(a.output_dir) / "search_status.json").read_text())
    print(json.dumps(st["calibration"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
