"""Per-image detection cut-offs from the search-stage z-score map (new; not part of v2.1).

Inspired by the extreme-value cut-off of Leopard-EM (Giammar et al., Acta Cryst D 2026) and
written independently. Every pixel of ``search_mip_scaled.mrc`` is the maximum of many
correlated Gaussian scores, so the background follows extreme-value statistics. Estimating
that law from the image itself gives a cut-off that adapts to the image (ice, crowding,
template) instead of assuming independent Gaussian trials.

Methods (all aim at ``n_false`` false detections per image):

* ``gev_bulk``: a generalised extreme-value (GEV) law fitted to all background pixels,
  cut-off from ``n_valid * S(c) = n_false``. Reported for comparison: the fit is dominated
  by the bulk of the distribution, and its tail is too heavy or too light for the cut-off.
* ``gpd_pixel`` / ``gpd_cluster`` (default): peaks-over-threshold. Only the upper tail
  (pixels above the ``tail_q`` quantile, 0.99 by default) is modelled, with a generalised
  Pareto law, which is the limiting law of exceedances for any extreme-value parent. The
  tail model is ``S(x) = (1 - tail_q) * GPD_sf(x - u)``.

Two further refinements:

1. **Peak masking.** True particles sit in the upper tail and would inflate it. Discs (one
   particle radius) are masked around clusters that the current model calls very unlikely
   as background (above the level expected to be reached by ``mask_false`` = 0.05 clusters
   per image), and the tail is refitted until the mask is stable. Masking at the cut-off
   itself would remove ordinary background extremes too and drive the cut-off down. The
   first pass masks robust outliers (median + 6 robust SD) so that a crowded image does not
   start from a tail made of particles; each later mask is rebuilt from the current model,
   so background pixels masked in the first pass come back.
2. **Clusters, not pixels.** One false detection is a connected cluster of exceeding pixels.
   The number of clusters per exceeding pixel, r(x) <= 1, is measured on the image in the
   well-populated range and extrapolated with log r = a + b log x (cluster area shrinks as the
   level rises). ``gpd_cluster`` solves ``n_valid * S(c) * r(c) = n_false``.

When decoy searches are given, the empirical decoy cut-off (clusters per image) is reported
as well; it is the reference the model cut-offs should be checked against.

The cut-offs apply to the search-stage score (``z_search`` in particles_qc.csv), not to the
refined/optimised ``score_final``.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import brentq
from scipy.stats import genextreme, genpareto

from . import io as sio

EIGHT = np.ones((3, 3))
METHODS = ("erfc_v21", "gev_bulk", "gpd_pixel", "gpd_cluster", "decoy")


def n_clusters(z: np.ndarray, x: float, mask: np.ndarray | None = None) -> int:
    """Number of connected clusters (8-connectivity) of pixels above x, inside mask."""
    b = z > x
    if mask is not None:
        b &= mask
    return int(ndimage.label(b, structure=EIGHT)[1])


def disc_mask(z: np.ndarray, x: float, radius: float, valid: np.ndarray) -> np.ndarray:
    """valid pixels farther than ``radius`` from any cluster above x."""
    above = (z > x) & valid
    if not above.any():
        return valid.copy()
    dist = ndimage.distance_transform_edt(~above)
    return valid & (dist > radius)


def fit_gev(values: np.ndarray, max_samples: int = 200_000, seed: int = 0):
    """GEV fit to the values (a random subset above max_samples).
    Returns (frozen distribution, (shape, loc, scale))."""
    v = np.asarray(values, dtype=np.float64)
    if v.size > max_samples:
        v = np.random.default_rng(seed).choice(v, max_samples, replace=False)
    shape, loc, scale = genextreme.fit(v)
    return genextreme(shape, loc=loc, scale=scale), (float(shape), float(loc), float(scale))


def cluster_ratio_fit(z: np.ndarray, mask: np.ndarray, min_clusters: int = 20, n_points: int = 25) -> dict:
    """Fit log(clusters/pixels) = a + b log x over thresholds with >= min_clusters clusters."""
    v = z[mask]
    xs = np.quantile(v, np.linspace(0.90, 0.9995, n_points))
    xs = xs[xs > 0]
    pts = []
    for x in xs:
        npx = int((v > x).sum())
        nc = n_clusters(z, x, mask)
        if nc >= min_clusters and npx > 0:
            pts.append((x, nc / npx))
    if len(pts) < 3:
        return dict(a=0.0, b=0.0, n_points=len(pts), note="too few points; r(x) = 1 (pixel counting)")
    x, r = np.array(pts).T
    b, a = np.polyfit(np.log(x), np.log(r), 1)
    return dict(a=float(a), b=float(b), n_points=len(pts), x_range=[float(x.min()), float(x.max())])


def _r_of(x: float, fit: dict) -> float:
    """Clusters per exceeding pixel at threshold x, from the fit of cluster_ratio_fit (at most 1)."""
    return float(min(1.0, np.exp(fit["a"] + fit["b"] * np.log(max(x, 1e-6)))))


def _solve(fun, lo: float, hi: float) -> float:
    """Root of fun between lo and hi (NaN when fun does not change sign there)."""
    if fun(lo) * fun(hi) > 0:
        return float("nan")
    return float(brentq(fun, lo, hi, xtol=1e-4))


def fit_tail(values: np.ndarray, tail_q: float = 0.99):
    """Generalised Pareto fit to exceedances above the tail_q quantile -> (u, shape, scale)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size < 1000:
        raise ValueError(f"only {v.size} background pixels left for the tail fit; "
                         "use a smaller --radius or a decoy search")
    u = float(np.quantile(v, tail_q))
    y = v[v > u] - u
    shape, _, scale = genpareto.fit(y, floc=0)
    return u, float(shape), float(scale)


def tail_sf(x, u: float, shape: float, scale: float, tail_q: float):
    """Survival function of the tail model: (1 - tail_q) x GPD_sf(x - u)."""
    return (1 - tail_q) * genpareto.sf(np.asarray(x) - u, shape, 0, scale)


def _excess(x: float, n: float, tail: tuple, rfit: dict | None, target: float) -> float:
    """Expected background detections above x minus the target (clusters if rfit is given)."""
    e = n * tail_sf(x, *tail)
    return float(e * (_r_of(x, rfit) if rfit is not None else 1.0) - target)


def centre_mask(shape, centres, radius: float) -> np.ndarray:
    """True farther than ``radius`` from every centre (0-based row, col)."""
    seeds = np.zeros(shape, bool)
    for r, c in centres:
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            seeds[int(r), int(c)] = True
    if not seeds.any():
        return np.ones(shape, bool)
    return ndimage.distance_transform_edt(~seeds) > radius


def tail_cutoff(z: np.ndarray, valid: np.ndarray, n_false: float = 1.0, radius: float = 10.0,
                tail_q: float = 0.99, mask_false: float = 0.05, max_iter: int = 8, k0: float = 6.0,
                centres=None, min_keep: float = 0.05) -> dict:
    """Peaks-over-threshold cut-off for ``n_false`` background detections per image.

    Signal is kept out of the fit by discs of ``radius`` around the search's own refined
    candidates (``centres``, when given) and, iteratively, around clusters above the level at
    which ``mask_false`` background clusters are expected. In crowded images the discs could
    cover everything: the radius is then halved until at least ``min_keep`` of the valid
    pixels remain, and the radius used is reported."""
    v = z[valid]
    med = float(np.median(v))
    rsd = float(1.4826 * np.median(np.abs(v - med)))
    n = float(valid.sum())
    hi = float(z[valid].max()) + 20

    def build(level, r):
        """Pixels kept for the fit: valid, outside discs of radius r around the centres and
        around clusters above level."""
        m = valid.copy()
        if centres is not None and len(centres):
            m &= centre_mask(z.shape, centres, r)
        if level is not None:
            m &= disc_mask(z, level, r, valid)
        return m

    def guarded(level, r):
        """build(), halving r until at least min_keep of the valid pixels remain."""
        m = build(level, r)
        while m.sum() < min_keep * n and r > 2:
            r /= 2
            m = build(level, r)
        return m, r

    first = None if (centres is not None and len(centres)) else med + k0 * rsd
    mask, r_used = guarded(first, radius)
    hist = []
    for _ in range(max_iter):
        u, shape, scale = fit_tail(z[mask], tail_q)
        rfit = cluster_ratio_fit(z, mask)
        tail = (u, shape, scale, tail_q)
        c_px = _solve(partial(_excess, n=n, tail=tail, rfit=None, target=n_false), u, hi)
        c_cl = _solve(partial(_excess, n=n, tail=tail, rfit=rfit, target=n_false), u, hi)
        c_mask = _solve(partial(_excess, n=n, tail=tail, rfit=rfit, target=mask_false), u, hi)
        hist.append(dict(cutoff=c_cl, mask_level=c_mask, masked=int(n - mask.sum()), radius_px=r_used))
        if not np.isfinite(c_mask):
            break
        new, r_new = guarded(c_mask, radius)
        if np.array_equal(new, mask):
            break
        mask, r_used = new, r_new
    dist, params = fit_gev(z[mask])
    c_gev = _solve(lambda x: n * dist.sf(x) - n_false, float(np.median(z[mask])), hi)
    return dict(gpd_cluster=c_cl, gpd_pixel=c_px, gev_bulk=c_gev,
                tail=dict(tail_q=tail_q, u=u, shape=shape, scale=scale),
                gev_params=dict(shape=params[0], loc=params[1], scale=params[2]),
                cluster_ratio=rfit, iterations=hist, n_valid=int(n), n_masked=int(n - mask.sum()),
                mask_radius_used_px=float(r_used), n_centres=0 if centres is None else len(centres))


def decoy_cutoff(decoy_maps: list, n_false: float = 1.0, xs=None) -> dict:
    """Empirical cut-off: mean number of clusters per decoy image = n_false."""
    if xs is None:
        xs = np.linspace(3.0, 12.0, 361)
    per = np.array([[n_clusters(z, x, v) for x in xs] for z, v in decoy_maps], dtype=np.float64)
    mean = per.mean(axis=0)
    # scan from the top: the z map of a MIP has a raised background, and at low thresholds all
    # pixels merge into one connected region, so the cluster count falls again there
    bad = np.nonzero(mean > n_false)[0]
    if not bad.size:
        c = float(xs[0])
    elif bad[-1] == len(xs) - 1:
        c = float("nan")
    else:
        c = float(xs[bad[-1] + 1])
    return dict(decoy=c, n_decoy_images=len(decoy_maps), xs=xs.tolist(), mean_clusters=mean.tolist())


def model_radius_A(path) -> float | None:
    """Largest distance (Å) of the searched model's atoms (or of the density above 1% of the
    maximum, for a map) from its centre."""
    p = Path(path)
    if not p.is_file():
        return None
    if p.suffix.lower() in (".cif", ".mmcif", ".pdb"):
        from .structure import read_cif_file, read_pdb_file
        at = read_pdb_file(p) if p.suffix.lower() == ".pdb" else read_cif_file(p)
        xyz = at.xyz[:, np.isfinite(at.xyz).all(axis=0)]
        if xyz.shape[1] == 0:
            return None
        return float(np.linalg.norm(xyz - xyz.mean(axis=1, keepdims=True), axis=0).max())
    if p.suffix.lower() == ".mrc":
        v = np.asarray(sio.mr(p), dtype=np.float64)
        hdr = np.fromfile(p, dtype=np.int32, count=10)
        cella = float(np.fromfile(p, dtype=np.float32, count=13)[10])
        a = cella / hdr[7] if hdr[7] > 0 and cella > 0 else 1.0
        idx = np.argwhere(v > 0.01 * v.max())
        if not len(idx):
            return None
        return float(np.linalg.norm(idx - idx.mean(axis=0), axis=1).max() * a)
    return None


def particle_radius_px(out_dir, fallback: float = 10.0) -> tuple[float, str]:
    """Masking radius on the search grid from the searched model (search_status.json)."""
    st = Path(out_dir) / "search_status.json"
    if st.exists():
        js = json.loads(st.read_text())
        a = js.get("sampling_A")
        files = (js.get("provenance") or {}).get("input_sha256") or {}
        for key in ("structureFile", "modelFile"):
            path = (files.get(key) or {}).get("path")
            r = model_radius_A(path) if path else None
            if r and a:
                return r / float(a), f"{key} {Path(path).name}: {r:.0f} A at {float(a):.3f} A/px"
    return fallback, "fallback (model or sampling not found)"


def analysis_mask(out_dir, valid: np.ndarray, edge_px: float = 0.0, sd_floor: float = 0.5):
    """Pixels whose z-scores are usable for noise statistics (NEW).

    The search image is zero-padded to an FFT size and combine keeps all but a thin frame
    margin, so the z map contains the padding, where the orientation SD is near zero and the
    flat-fielded score is meaningless. Keep only the true image area (non-zero rows/columns of
    searchImage.mrc) shrunk by ``edge_px`` (partial template overlap at the image edge), and
    pixels whose orientation SD is at least ``sd_floor`` x the median."""
    out_dir = Path(out_dir)
    keep = valid.copy()
    info = dict(edge_px=float(edge_px), sd_floor=sd_floor, n_valid=int(valid.sum()))
    si = out_dir / "searchImage.mrc"
    if si.exists():
        img = sio.mr(si)
        if img.shape == valid.shape:
            # the padding is a constant (≈0 after mean subtraction, not exactly 0): image pixels
            # are those that differ from the corner value
            img = np.asarray(img, dtype=np.float64)
            dev = np.abs(img - img[0, 0]) > 1e-6 * (float(np.std(img)) or 1.0)
            rows, cols = np.nonzero(dev.any(axis=1))[0], np.nonzero(dev.any(axis=0))[0]
            e = int(np.ceil(edge_px))
            box = np.zeros_like(valid)
            box[rows.min() + e:rows.max() + 1 - e, cols.min() + e:cols.max() + 1 - e] = True
            keep &= box
            info["image_box"] = [int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())]
    sdf = out_dir / "search_SD.mrc"
    if sdf.exists():
        sd = np.asarray(sio.mr(sdf), dtype=np.float64)
        if sd.size % valid.size == 0:
            sd = sd.reshape(valid.shape[0], valid.shape[1], -1).mean(axis=2)
            med = float(np.median(sd[keep])) if keep.any() else float("nan")
            keep &= sd >= sd_floor * med
            info["sd_median"] = med
    info["n_kept"] = int(keep.sum())
    info["kept_fraction"] = round(info["n_kept"] / max(info["n_valid"], 1), 4)
    return keep, info


def load_z(out_dir, edge_px: float = 0.0, with_info: bool = False):
    """The search z-score map and the mask of pixels to analyse (with_info: also its description)."""
    z = sio.mr(Path(out_dir) / "search_mip_scaled.mrc").astype(np.float64)
    keep, info = analysis_mask(out_dir, z != 0, edge_px)
    return (z, keep, info) if with_info else (z, keep)


def search_centres(out_dir) -> list:
    """0-based search-grid positions of the refined candidates (particles.csv), if present."""
    f = Path(out_dir) / "particles.csv"
    if not f.exists():
        return []
    import csv
    with open(f) as fh:
        return [(int(round(float(r["row_search"]))) - 1, int(round(float(r["col_search"]))) - 1)
                for r in csv.DictReader(fh)]


def report(out_dir, n_false: float = 1.0, radius: float = 10.0, decoys=(), tail_q: float = 0.99) -> dict:
    """Every cutoff of one search (v2.1 threshold, GEV, tail fits, decoys) and the number of
    clusters observed above each."""
    out_dir = Path(out_dir)
    z, valid, info = load_z(out_dir, edge_px=radius, with_info=True)
    rep = dict(out_dir=str(out_dir), n_false=n_false, radius_px=radius, analysis_mask=info)
    centres = search_centres(out_dir)
    st = out_dir / "search_status.json"
    if st.exists():
        rep["erfc_v21"] = json.loads(st.read_text()).get("gaussian_threshold")
    rep.update(tail_cutoff(z, valid, n_false, radius, tail_q, centres=centres))
    if decoys:
        rep.update(decoy_cutoff([load_z(d, edge_px=radius) for d in decoys], n_false))
    rep["observed_clusters"] = {k: n_clusters(z, rep[k], valid) for k in METHODS
                                if isinstance(rep.get(k), float) and np.isfinite(rep[k])}
    return rep


def plot(rep: dict, z: np.ndarray, valid: np.ndarray, path) -> None:
    """Survival curve of the z-scores with the fitted models and the cutoffs, saved to path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p, t = rep["gev_params"], rep["tail"]
    dist = genextreme(p["shape"], loc=p["loc"], scale=p["scale"])
    v = np.sort(z[valid])
    xs = np.linspace(v[len(v) // 2], v[-1] + 0.5, 300)
    surv = v.size - np.searchsorted(v, xs, side="right")
    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    ax.semilogy(xs, np.maximum(surv, 0.5), color="#222222", lw=1.6, label="pixels above x (image)")
    ax.semilogy(xs, v.size * dist.sf(xs), color="#9DB9D5", lw=1.2, label="GEV, whole distribution")
    xt = xs[xs >= t["u"]]
    ax.semilogy(xt, v.size * tail_sf(xt, t["u"], t["shape"], t["scale"], t["tail_q"]), color="#2878B5", lw=1.6,
                label=f"tail model (GPD above q={t['tail_q']})")
    for key, col in (("erfc_v21", "#888888"), ("gev_bulk", "#9DB9D5"), ("gpd_pixel", "#2878B5"),
                     ("gpd_cluster", "#D18A20"), ("decoy", "#B03A2E")):
        if isinstance(rep.get(key), float) and np.isfinite(rep[key]):
            ax.axvline(rep[key], color=col, ls="--", lw=1, label=f"{key} = {rep[key]:.2f}")
    ax.set_xlabel("search-stage z-score")
    ax.set_ylabel("count")
    ax.set_ylim(0.5, v.size * 1.5)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-image detection cut-offs (GEV, decoy) for a finished search")
    ap.add_argument("output_dir")
    ap.add_argument("--n-false", type=float, default=1.0, help="accepted false detections per image")
    ap.add_argument("--radius", type=float, default=None,
                    help="particle radius on the search grid (pixels); default: from the searched model")
    ap.add_argument("--decoy", nargs="*", default=[], help="decoy search output folders (same images)")
    ap.add_argument("--tail-q", type=float, default=0.99, help="quantile above which the tail is modelled")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(argv)
    radius, how = (a.radius, "given") if a.radius is not None else particle_radius_px(a.output_dir)
    print(f"mask radius {radius:.1f} px ({how}); image edge band excluded: {radius:.0f} px")
    rep = report(a.output_dir, a.n_false, radius, a.decoy, a.tail_q)
    rep["radius_source"] = how
    out = Path(a.output_dir) / "cutoffs.json"
    out.write_text(json.dumps({k: v for k, v in rep.items() if k not in ("xs", "mean_clusters")}, indent=1))
    if a.plot:
        z, valid = load_z(a.output_dir, edge_px=radius)
        plot(rep, z, valid, Path(a.output_dir) / "cutoffs.png")
    keys = [k for k in METHODS if rep.get(k) is not None]
    if rep.get("mask_radius_used_px", radius) < radius:
        print(f"note: crowded image; particle discs shrunk to {rep['mask_radius_used_px']:.1f} px "
              f"to keep >= 5% of the pixels for the noise fit")
    m = rep["analysis_mask"]
    print(f"pixels used: {m['n_kept']} of {m['n_valid']} ({100 * m['kept_fraction']:.1f}%; padding, edge band, low SD removed)")
    print("  ".join(f"{k}={rep[k]:.2f}" for k in keys), f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
