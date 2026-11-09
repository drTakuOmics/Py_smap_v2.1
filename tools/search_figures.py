#!/usr/bin/env python
"""Figures for one full-micrograph search: detected poses on the micrograph, the whole z-score
map and the z-score landscape with the cut-off.

Inputs are read from the search output folder (``search_mip_scaled.mrc``, ``particles.csv``,
``search_status.json``, the template potential ``*_SP.mrc``); the micrograph path comes from
``search_status.json`` unless ``--micrograph`` is given.

Outputs in ``--out`` (never overwritten): ``poses.png`` (micrograph, binned and low-passed, with
the template surface rendered at every detection with final score >= cutoff), ``poses_pairs.png``
(raw | overlay for the top candidates), ``zmap.png`` (search z-score over the whole image, cut-off
contour, detections circled), ``landscape.png`` (3-D z-score landscape with the cut-off plane),
``summary.png`` (poses | z map | landscape) and ``figures.json`` (inputs and display settings).

Surfaces: the potential is low-passed and resampled to ``--render-A``, rotated to each pose
(V(Rᵀu), the convention of the search templates: the projection of the rotated volume along z
is the template), thresholded at the level enclosing ``--mass-frac`` of its density and shown as
a depth-shaded height map seen along the beam, centred at x_orig0_est, y_orig0_est.

    python tools/search_figures.py RESULT_DIR --cutoff 7.25 --label "LSU (6EK0)" --out FIG_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # repo root (smap) when run as a script

import argparse
import csv
import json
import math

import numpy as np

from smap import io as sio

BASE = "#F0B43C"


# ------------------------------------------------------------------ inputs

def read_particles(out_dir: Path) -> list:
    f = out_dir / "particles.csv"
    if not f.exists():
        raise SystemExit(f"{f} missing (search without refinement?)")
    rows = []
    for r in csv.DictReader(f.open()):
        q = np.array([[float(r[f"q_opt_{i}{j}"]) for j in range(3)] for i in range(3)])
        rows.append(dict(id=int(r["num_id"]), score=float(r["score_final"]),
                         x=float(r["x_orig0_est"]), y=float(r["y_orig0_est"]),
                         r=float(r["row_search"]), c=float(r["col_search"]), q=q))
    return rows


def status(out_dir: Path) -> dict:
    f = out_dir / "search_status.json"
    return json.loads(f.read_text()) if f.exists() else {}


def par_value(text: str, key: str):
    for ln in (text or "").splitlines():
        t = ln.split()
        if t and t[0] == key and "#" not in ln and len(t) > 1:
            return t[1]
    return None


def find_volume(out_dir: Path, st: dict) -> Path:
    p = (st.get("provenance", {}).get("input_sha256", {}).get("modelFile", {}) or {}).get("path")
    if p and Path(p).suffix == ".mrc" and Path(p).exists():
        return Path(p)
    c = sorted(out_dir.glob("*_SP.mrc"))
    if not c:
        raise SystemExit(f"no *_SP.mrc in {out_dir}; pass --volume")
    return c[0]


def voxel_size(path: Path, default: float) -> float:
    try:
        import mrcfile
        with mrcfile.open(path, permissive=True, header_only=True) as m:
            v = float(m.voxel_size.x)
        return v if v > 0 else default
    except Exception:
        return default


def bin_mean(a: np.ndarray, b: int) -> np.ndarray:
    n0, n1 = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    return a[:n0, :n1].reshape(n0 // b, b, n1 // b, b).mean(axis=(1, 3))


def bin_max(a: np.ndarray, b: int) -> np.ndarray:
    n0, n1 = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN bins stay NaN
        return np.nanmax(a[:n0, :n1].reshape(n0 // b, b, n1 // b, b), axis=(1, 3))


# ------------------------------------------------------------------ surfaces

def prepare_volume(V: np.ndarray, a: float, render_A: float, mass_frac: float, sigma_A: float = 1.5):
    """Background-subtracted volume at render_A voxels, its iso level and box.

    The envelope (how much is enclosed) is set on a smooth copy (4 Å Gaussian: the level that
    encloses ``mass_frac`` of the density); the rendered surface is the sharper copy (``sigma_A``)
    at the level enclosing the same volume, so detail does not change the particle size.
    The box is cropped to the particle; detached specks are removed."""
    from scipy.ndimage import binary_dilation, gaussian_filter, label, zoom
    V = np.asarray(V, dtype=np.float32)
    shell = np.concatenate([V[0].ravel(), V[-1].ravel(), V[:, 0].ravel(), V[:, -1].ravel(),
                            V[:, :, 0].ravel(), V[:, :, -1].ravel()])
    V = V - np.median(shell)
    f = a / render_A
    if f < 1:
        V = zoom(gaussian_filter(V, 0.5 / f), f, order=1)
    vox = a / f if f < 1 else a
    Vs = gaussian_filter(V, 4.0 / vox)
    pos = np.sort(Vs[Vs > 0].ravel())[::-1]
    cum = np.cumsum(pos, dtype=np.float64)
    n_in = int(min(np.searchsorted(cum, mass_frac * cum[-1]) + 1, pos.size))
    V = gaussian_filter(V, max(sigma_A / vox, 0.5))
    level = float(np.partition(V.ravel(), V.size - n_in)[V.size - n_in])
    lab, n = label(V > level)
    if n > 1:                                                    # keep the largest component
        big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
        V = np.where(binary_dilation(lab == big, iterations=2), V, np.minimum(V, level - 1e-6))
    c = V.shape[0] // 2
    idx = np.argwhere(V > level)
    r = int(np.ceil(np.sqrt(((idx - c) ** 2).sum(axis=1)).max())) + 3 if idx.size else c
    r = min(r, c)
    V = V[c - r:c + r, c - r:c + r, c - r:c + r]                 # centre stays at N//2
    return np.ascontiguousarray(V), level, vox


def rotate_volume(V: np.ndarray, R: np.ndarray) -> np.ndarray:
    """W(u) = V(Rᵀ u) on the same grid; arrays in the port layout [y, x, z] about N//2."""
    from scipy.ndimage import map_coordinates
    n = V.shape[0]
    c = n // 2
    g = np.arange(n) - c
    uy, ux, uz = np.meshgrid(g, g, g, indexing="ij")
    u = np.stack([ux.ravel(), uy.ravel(), uz.ravel()]).astype(np.float32)   # (x, y, z)
    p = (np.asarray(R, dtype=np.float64).T @ u).astype(np.float32)
    W = map_coordinates(V, [p[1] + c, p[0] + c, p[2] + c], order=1, cval=0.0)
    return W.reshape(n, n, n)


def _ambient_occlusion(H: np.ndarray, floor: float, radius: int = 24, n_dir: int = 16) -> np.ndarray:
    """Horizon-based ambient occlusion of a height map (1 = open, 0 = fully occluded)."""
    from scipy.ndimage import gaussian_filter
    R = radius
    H = gaussian_filter(H, 1.0)
    P = np.pad(H, R, constant_values=floor)
    ny, nx = H.shape
    steps = sorted({int(round(v)) for v in np.geomspace(2, R, 6)})
    occ = np.zeros_like(H)
    for k in range(n_dir):
        t = 2 * np.pi * (k + 0.5) / n_dir
        best = np.zeros_like(H)
        for st in steps:
            dx, dy = int(round(np.cos(t) * st)), int(round(np.sin(t) * st))
            if dx == 0 and dy == 0:
                continue
            Hs = P[R + dy:R + dy + ny, R + dx:R + dx + nx]
            best = np.maximum(best, (Hs - H) / np.hypot(dx, dy))
        occ += best / np.sqrt(1 + best ** 2)                         # sin(horizon elevation)
    return np.clip(1 - occ / n_dir, 0, 1)


def radial_limits(V: np.ndarray, level: float, vox: float, n_dir: int = 1500) -> tuple:
    """Range (A) of the outer-surface radius, as seen from outside along n_dir directions
    (0.5th percentile to maximum, so small spikes keep the top colour); the colour scale of
    ``--color-by radial``."""
    from scipy.ndimage import map_coordinates
    n = V.shape[0]
    c = n // 2
    k = np.arange(n_dir) + 0.5
    phi, cos_t = np.pi * (1 + 5 ** 0.5) * k, 1 - 2 * k / n_dir         # Fibonacci sphere
    sin_t = np.sqrt(1 - cos_t ** 2)
    d = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)
    r = np.arange(c - 1, 0, -0.5)                                       # outside -> centre (voxels)
    p = d[:, None, :] * r[None, :, None] + c                            # (x, y, z) voxel coordinates
    vals = map_coordinates(V, [p[..., 1].ravel(), p[..., 0].ravel(), p[..., 2].ravel()], order=1)
    inside = vals.reshape(n_dir, r.size) > level
    hit = inside.any(axis=1)
    r_out = r[np.argmax(inside, axis=1)][hit] * vox
    return (float(np.percentile(r_out, 0.5)), float(r_out.max())) if r_out.size else (0.0, c * vox)


def render(W: np.ndarray, level: float, color=BASE, lighting: str = "soft",
           alpha: float = 0.95, voxel_A: float = 1.5, radial=None) -> np.ndarray:
    """RGBA rendering of {W > level} seen from +z (rows y, cols x) from its height map.

    lighting: ``soft`` (ambient occlusion only, as ChimeraX 'lighting soft'), ``simple`` (one
    diffuse key light + ambient, no highlights), ``flat`` (ambient only). Thin dark silhouette.
    radial: (r_min_A, r_max_A, colormap) colours the surface by its distance from the volume
    centre (N//2), as capsids are usually shown; otherwise the single ``color``."""
    from matplotlib.colors import to_rgb
    from scipy.ndimage import binary_erosion, gaussian_filter
    M = W > level
    hit = M.any(axis=2)
    rgba = np.zeros(hit.shape + (4,))
    if not hit.any():
        return rgba
    n = W.shape[2]
    top = n - 1 - np.argmax(M[:, :, ::-1], axis=2)                 # highest z inside
    k = np.clip(top, 1, n - 2)                                       # sub-voxel surface height
    rr, cc = np.nonzero(hit)
    w0, w1 = W[rr, cc, k[rr, cc]], W[rr, cc, np.minimum(k[rr, cc] + 1, n - 1)]
    D = np.full(hit.shape, np.nan)
    D[rr, cc] = top[rr, cc] + np.clip((w0 - level) / np.where(w0 - w1 == 0, 1, w0 - w1), 0, 1)
    lo = float(np.nanmin(D))
    floor = lo - 0.25 * (float(np.nanmax(D)) - lo)
    Df = gaussian_filter(np.where(hit, D, floor), 0.6)
    gy, gx = np.gradient(Df)
    nrm = np.stack([-gx, -gy, np.ones_like(gx)])
    nrm /= np.linalg.norm(nrm, axis=0)
    L = np.array([-0.35, -0.45, 0.82])
    L /= np.linalg.norm(L)
    lam = np.clip(np.tensordot(L, nrm, axes=1), 0, 1)
    if lighting == "soft":
        ao = _ambient_occlusion(Df, floor, radius=max(8, int(round(36.0 / voxel_A))))
        shade = 0.32 + 0.75 * ao + 0.08 * lam
    elif lighting == "simple":
        shade = 0.45 + 0.60 * lam
    elif lighting == "flat":
        shade = np.ones_like(lam)
    else:
        raise ValueError(f"lighting {lighting!r}: soft, simple or flat")
    if radial is None:
        base = np.broadcast_to(np.array(to_rgb(color)), hit.shape + (3,))
    else:
        import matplotlib
        r0, r1, cmap = radial
        c0 = n // 2
        yy, xx = np.indices(hit.shape)
        rad = np.sqrt((xx - c0) ** 2 + (yy - c0) ** 2 + (np.nan_to_num(D, nan=c0) - c0) ** 2) * voxel_A
        base = matplotlib.colormaps[cmap](np.clip((rad - r0) / max(r1 - r0, 1e-6), 0, 1))[..., :3]
        base = 0.08 + 0.84 * base if cmap == "turbo" else base               # avoid the darkest ends
    col = base * np.clip(shade, 0.0, 1.15)[..., None]
    rgba[..., :3] = np.clip(col, 0, 1)
    edge = hit & ~binary_erosion(hit)
    rgba[edge, :3] *= 0.35
    rgba[..., 3] = np.where(hit, alpha, 0.0)
    return rgba


# ------------------------------------------------------------------ figure helpers

def style():
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.edgecolor": "#888888",
                         "axes.labelcolor": "#333333", "xtick.color": "#555555",
                         "ytick.color": "#555555", "savefig.dpi": 200, "font.family": "DejaVu Sans"})


def show_micrograph(ax, img_disp, extent):
    lo, hi = np.percentile(img_disp, [0.5, 99.5])
    ax.imshow(img_disp, cmap="gray", vmin=lo, vmax=hi, extent=extent, origin="upper",
              interpolation="bilinear")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")


def draw_poses(ax, sprites, parts, box_A):
    for d in parts:
        cx, cy = d["xA"], d["yA"]
        h = box_A / 2
        ax.imshow(sprites[d["id"]], extent=(cx - h, cx + h, cy + h, cy - h), origin="upper",
                  interpolation="antialiased", interpolation_stage="rgba", zorder=3)


def scale_bar(ax, length_A, extent, color="white"):
    x1, x2, y1, y2 = extent[0], extent[1], extent[3], extent[2]
    w, hgt = x2 - x1, y2 - y1
    xs = x2 - 0.05 * w - length_A
    ys = y2 - 0.05 * hgt
    ax.plot([xs, xs + length_A], [ys, ys], color=color, lw=2.5, solid_capstyle="butt", zorder=5)
    ax.text(xs + length_A / 2, ys - 0.015 * hgt, f"{length_A / 10:g} nm", color=color, ha="center",
            va="bottom", fontsize=8, zorder=5)


def save(fig, path: Path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("result", help="search output folder")
    ap.add_argument("--cutoff", type=float, required=True, help="final-score cut-off for detections")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="template")
    ap.add_argument("--micrograph", default=None)
    ap.add_argument("--volume", default=None, help="template potential .mrc (default: *_SP.mrc)")
    ap.add_argument("--color", default=BASE)
    ap.add_argument("--color-by", default="uniform", choices=("uniform", "radial"),
                    help="radial: colour by distance from the particle centre (capsids, cages)")
    ap.add_argument("--cmap", default="turbo", help="colormap for --color-by radial")
    ap.add_argument("--lighting", default="soft", choices=("soft", "simple", "flat"),
                    help="soft = ambient occlusion (ChimeraX 'lighting soft'); simple = one diffuse light")
    ap.add_argument("--render-A", type=float, default=1.5, help="surface voxel (Å); 3-4 for speed")
    ap.add_argument("--surface-sigma-A", type=float, default=1.5,
                    help="Gaussian low-pass of the rendered surface (Å); larger = smoother")
    ap.add_argument("--overview-render-A", type=float, default=3.0,
                    help="surface voxel for the whole-micrograph panels (Å)")
    ap.add_argument("--overview-sigma-A", type=float, default=3.0)
    ap.add_argument("--zmap-bin-A", type=float, default=12.0, help="z-map display bin (max-binning)")
    ap.add_argument("--mass-frac", type=float, default=0.85)
    ap.add_argument("--display-bin-A", type=float, default=4.0, help="micrograph display pixel")
    ap.add_argument("--landscape-bin-A", type=float, default=12.0)
    ap.add_argument("--n-pairs", type=int, default=3)
    ap.add_argument("--pair-half-A", type=float, default=550.0)
    ap.add_argument("--elev", type=float, default=32)
    ap.add_argument("--azim", type=float, default=-62)
    a = ap.parse_args(argv)

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new --out (figures are never overwritten)")
    res = Path(a.result)
    st = status(res)
    text = st.get("provenance", {}).get("params_text", "")
    a_orig = float(par_value(text, "aPerPix") or 0) or None
    a_search = float(st.get("sampling_A") or 0) or None
    img_path = Path(a.micrograph or st.get("provenance", {}).get("input_sha256", {})
                    .get("imageFile", {}).get("path", ""))
    if not img_path.is_file():
        raise SystemExit(f"micrograph {img_path} not found; pass --micrograph")
    if a_orig is None or a_search is None:
        raise SystemExit("search_status.json lacks aPerPix / sampling_A")
    vol_path = Path(a.volume) if a.volume else find_volume(res, st)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource, Normalize
    from scipy.ndimage import gaussian_filter

    from smap.cutoff import load_z
    style()

    # ---- data
    img = sio.mr(img_path).astype(np.float64)
    H, W = img.shape
    b = max(1, int(round(a.display_bin_A / a_orig)))
    disp = gaussian_filter(bin_mean(img, b), 1.0)
    ext = (0, disp.shape[1] * b * a_orig, disp.shape[0] * b * a_orig, 0)
    b2 = max(1, int(round(2.0 / a_orig)))                          # ~2 Å for the zoomed pairs
    disp2 = gaussian_filter(bin_mean(img, b2), 2.0)
    ext2 = (0, disp2.shape[1] * b2 * a_orig, disp2.shape[0] * b2 * a_orig, 0)

    z, keep = load_z(res)
    zm = np.where(keep, z, np.nan)
    ratio = a_search / a_orig
    cs = np.array(z.shape) // 2 + 1
    co = np.array([H, W]) // 2 + 1

    def grid_to_A(r, c):          # 1-based search grid -> Å in the micrograph (zero-based px * a)
        y = (np.asarray(r, float) - cs[0]) * ratio + co[0] - 1
        x = (np.asarray(c, float) - cs[1]) * ratio + co[1] - 1
        return x * a_orig, y * a_orig

    x0A, y0A = grid_to_A(0.5, 0.5)
    x1A, y1A = grid_to_A(z.shape[0] + 0.5, z.shape[1] + 0.5)
    zext = (x0A, x1A, y1A, y0A)

    parts = read_particles(res)
    for d in parts:
        d["xA"], d["yA"] = d["x"] * a_orig, d["y"] * a_orig
        ri, ci = int(round(d["r"])) - 1, int(round(d["c"])) - 1
        d["z_search"] = float(z[ri, ci]) if 0 <= ri < z.shape[0] and 0 <= ci < z.shape[1] else float("nan")
    det = sorted([d for d in parts if d["score"] >= a.cutoff], key=lambda d: -d["score"])
    below = [d for d in parts if d["score"] < a.cutoff]

    # ---- surfaces
    a_vol = voxel_size(vol_path, a_orig)
    vol = sio.mr(vol_path)
    # close-ups: fine surface; whole micrograph: coarser surface (no sub-pixel detail to alias)
    V, level, vox = prepare_volume(vol, a_vol, a.render_A, a.mass_frac, a.surface_sigma_A)
    box_A = V.shape[0] * vox
    rlim = radial_limits(V, level, vox) if a.color_by == "radial" else None
    rad = (*rlim, a.cmap) if rlim else None
    sprites = {d["id"]: render(rotate_volume(V, d["q"]), level, a.color, a.lighting, voxel_A=vox, radial=rad)
               for d in det}
    Vo, level_o, vox_o = prepare_volume(vol, a_vol, a.overview_render_A, a.mass_frac, a.overview_sigma_A)
    box_o = Vo.shape[0] * vox_o
    sprites_o = {d["id"]: render(rotate_volume(Vo, d["q"]), level_o, a.color, a.lighting, voxel_A=vox_o,
                                 radial=rad) for d in det}

    def radial_bar(fig, ax, orientation="vertical"):
        if rlim:
            from matplotlib.cm import ScalarMappable
            sm = ScalarMappable(norm=Normalize(*rlim), cmap=a.cmap)
            if orientation == "vertical":
                fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.015, label="radius (Å)")
            else:                     # below the panel: keeps the neighbouring panel's axis label clear
                fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.09, aspect=40,
                             label="radius (Å)")
    del vol

    out.mkdir(parents=True)
    title = f"{a.label}: {len(det)} detections with final score ≥ {a.cutoff:g} ({len(parts)} refined)"

    # ---- poses
    fig, ax = plt.subplots(figsize=(8.2, 8.2 * disp.shape[0] / disp.shape[1]))
    show_micrograph(ax, disp, ext)
    draw_poses(ax, sprites_o, det, box_o)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    scale_bar(ax, 1000.0, ext)
    ax.set_title(title)
    radial_bar(fig, ax)
    save(fig, out / "poses.png")
    plt.close(fig)

    # ---- raw | overlay pairs
    npair = min(a.n_pairs, len(det))
    if npair:
        fig, axs = plt.subplots(npair, 2, figsize=(8, 4 * npair), squeeze=False)
        hpx = a.pair_half_A
        for k, d in enumerate(det[:npair]):
            for j in range(2):
                axx = axs[k, j]
                show_micrograph(axx, disp2, ext2)
                if j:
                    draw_poses(axx, sprites, det, box_A)
                cx = min(max(d["xA"], ext[0] + hpx), ext[1] - hpx)      # window inside the image
                cy = min(max(d["yA"], ext[3] + hpx), ext[2] - hpx)
                axx.set_xlim(cx - hpx, cx + hpx)
                axx.set_ylim(cy + hpx, cy - hpx)
                axx.set_xticks([])
                axx.set_yticks([])
                axx.set_xlabel("")
                axx.set_ylabel("")
            axs[k, 0].set_ylabel(f"#{d['id']}  score {d['score']:.1f}")
        axs[0, 0].set_title("micrograph")
        axs[0, 1].set_title(f"{a.label} poses")
        fig.tight_layout()
        save(fig, out / "poses_pairs.png")
        plt.close(fig)

    # ---- z-score map
    vals = zm[np.isfinite(zm)]
    zb = max(1, int(round(a.zmap_bin_A / a_search)))
    zdisp = bin_max(zm, zb)                     # max-binning keeps every peak height
    cmap = matplotlib.colormaps["magma"].with_extremes(bad="#d9d9d9")
    vlo = float(np.nanmedian(zdisp))
    vhi = max(float(vals.max()), a.cutoff + 1)

    def zpanel(axx, cbar=True):
        im = axx.imshow(zdisp, cmap=cmap, vmin=vlo, vmax=vhi, extent=zext, origin="upper",
                        interpolation="nearest")
        yy = np.linspace(zext[3], zext[2], zdisp.shape[0])
        xx = np.linspace(zext[0], zext[1], zdisp.shape[1])
        axx.contour(xx, yy, np.nan_to_num(zdisp, nan=0.0), levels=[a.cutoff], colors="#7fe3ff",
                    linewidths=0.6)
        for d in det:
            axx.plot(d["xA"], d["yA"], "o", mfc="none", mec="white", ms=7, mew=0.8)
        for d in below:
            axx.plot(d["xA"], d["yA"], "o", mfc="none", mec="#9a9a9a", ms=5, mew=0.6)
        axx.set_xlim(ext[0], ext[1])
        axx.set_ylim(ext[2], ext[3])
        axx.set_xlabel("x (Å)")
        axx.set_ylabel("y (Å)")
        if cbar:
            cb = plt.colorbar(im, ax=axx, shrink=0.8, pad=0.02)
            cb.set_label("search z-score (flat-fielded SNR)")
            cb.ax.axhline(a.cutoff, color="#7fe3ff", lw=1.2)
        return im

    fig, ax = plt.subplots(figsize=(8.8, 8.2 * disp.shape[0] / disp.shape[1]))
    zpanel(ax)
    ax.set_title(f"{a.label}: search z-score, whole micrograph (cyan: z = {a.cutoff:g}; "
                 f"white circles: final score ≥ {a.cutoff:g})", fontsize=9)
    save(fig, out / "zmap.png")
    plt.close(fig)

    # ---- landscape
    lb = max(1, int(round(a.landscape_bin_A / a_search)))
    G = bin_max(zm, lb)
    ny, nx = G.shape
    xs = np.linspace(zext[0], zext[0] + nx * lb * a_search, nx)
    ys = np.linspace(zext[3], zext[3] + ny * lb * a_search, ny)
    floor = math.floor(float(np.nanmedian(G)))
    top = max(math.ceil(float(np.nanmax(G))), a.cutoff + 1)
    Z = np.where(np.isfinite(G), np.clip(G, floor, top), floor)
    cm = matplotlib.colormaps["turbo"]
    rgb = LightSource(azdeg=315, altdeg=45).shade(Z, cmap=cm, norm=Normalize(floor - 0.3 * (top - floor), top),
                                                  vert_exag=0.05, blend_mode="soft")
    rgb[~np.isfinite(G)] = (0.92, 0.92, 0.92, 1.0)

    def lpanel(axx):
        X, Y = np.meshgrid(xs, ys)
        axx.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False,
                         shade=False)
        axx.plot_surface(X[[0, -1]][:, [0, -1]], Y[[0, -1]][:, [0, -1]], np.full((2, 2), a.cutoff),
                         color="#4a90d9", alpha=0.18, linewidth=0, shade=False)
        for d in det[:12]:
            axx.text(d["xA"], d["yA"], min(d["z_search"], top) + 0.04 * (top - floor), f"{d['id']}",
                     fontsize=6, color="#222222", ha="center", zorder=10)
        axx.set(xlim=(xs[0], xs[-1]), ylim=(ys[-1], ys[0]), zlim=(floor, top))
        axx.set_xlabel("x (Å)", labelpad=2)
        axx.set_ylabel("y (Å)", labelpad=2)
        axx.set_zlabel("z-score", labelpad=0)
        axx.view_init(elev=a.elev, azim=a.azim)
        axx.set_box_aspect((xs[-1] - xs[0], ys[-1] - ys[0], 0.35 * max(xs[-1] - xs[0], ys[-1] - ys[0])))
        for axis in (axx.xaxis, axx.yaxis, axx.zaxis):
            axis.pane.set_facecolor("white")
            axis.pane.set_edgecolor("#dddddd")
        axx.tick_params(labelsize=7, pad=0)

    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(projection="3d")
    lpanel(ax)
    ax.set_title(f"{a.label}: z-score landscape ({a.landscape_bin_A:g} Å max-binning); "
                 f"plane at {a.cutoff:g}; labels = 12 top detections", fontsize=9)
    save(fig, out / "landscape.png")
    plt.close(fig)

    # ---- summary (README)
    fig = plt.figure(figsize=(19, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.12, 1.35], wspace=0.16)
    ax1 = fig.add_subplot(gs[0, 0])
    show_micrograph(ax1, disp, ext)
    draw_poses(ax1, sprites_o, det, box_o)
    ax1.set_xlim(ext[0], ext[1])
    ax1.set_ylim(ext[2], ext[3])
    scale_bar(ax1, 1000.0, ext)
    ax1.set_title(f"{a.label}: {len(det)} poses (final score ≥ {a.cutoff:g})")
    radial_bar(fig, ax1, "horizontal")
    ax2 = fig.add_subplot(gs[0, 1])
    zpanel(ax2)
    ax2.set_title(f"search z-score (cyan contour: {a.cutoff:g})")
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    lpanel(ax3)
    ax3.set_title(f"z-score landscape, plane at {a.cutoff:g}")
    save(fig, out / "summary.png")
    plt.close(fig)

    meta = dict(result=str(res), micrograph=str(img_path), volume=str(vol_path), volume_voxel_A=a_vol, cutoff=a.cutoff,
                label=a.label, n_refined=len(parts), n_detected=len(det), a_orig=a_orig, a_search=a_search,
                render_A=vox, surface_sigma_A=a.surface_sigma_A, overview_render_A=vox_o,
                overview_sigma_A=a.overview_sigma_A, zmap_bin_A=zb * a_search, lighting=a.lighting,
                color_by=a.color_by, radial_limits_A=rlim,
                mass_frac=a.mass_frac, iso_level=level, display_bin_px=b,
                landscape_bin_px=lb, analysis_mask=float(np.mean(keep)),
                detections=[dict(id=d["id"], score=d["score"], x_A=round(d["xA"], 1), y_A=round(d["yA"], 1),
                                 z_search=round(d["z_search"], 3)) for d in det])
    (out / "figures.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}: {len(det)} of {len(parts)} refined candidates >= {a.cutoff:g}; "
          f"surfaces at {vox:g} A, iso level {level:.3g} ({a.mass_frac:g} of density)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
