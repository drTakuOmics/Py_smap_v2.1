#!/usr/bin/env python3
"""Z-score correlograms (2DTM peak landscapes) for RNF213 tiles searched with the port.

Uses the full-image maps the port writes for every tile (v2.1 never saved them):
search_mip_scaled.mrc (pixelwise z-score = flat-fielded SMAP SNR of the best orientation,
before refinement) and optionally search_pval.mrc (-ln p) or search_mip.mrc (raw CC).

Figures (PNG 300 dpi + PDF + SVG, JSON sidecar; an existing --out is never overwritten):
  overview          top view of the whole tile: z-score heatmap, cutoff contour, numbered
                    candidates (reference picks / matches if --reference is given)
  candidate_NN      one candidate: 3-D z-score landscape (Leopard-EM / HRTM style) with a
                    translucent cutoff plane and a projected floor map, the top view with
                    the cutoff contour, and x/y profiles through the peak. With --compare,
                    a second row shows the same micrograph region from another search
                    (e.g. full vs domain-omission anchor) on a shared z scale.
  gallery           small-multiple landscapes of all candidates, sorted by final score
  candidates.csv    candidate id, micrograph x/y (px and A), search z, -ln p, final score

Coordinates: map pixel (row, col) on the search grid -> zero-based tile pixel via the same
conversion as rnf213_workflow_v3.py (search-grid centre, aPerPix_search / aPerPix), plus the
tile offset from tile_offsets.csv -> micrograph x/y (px) -> Angstrom (x aPerPix).

usage (from the project root, port environment):
  python ../Py_smap_v2.1/tools/rnf213_correlogram.py --tile tiles_3p8_port/tile_0003 \\
         [--label "Full RNF213"] [--compare batch_noE3_3_port/searches/image_0001/tile_0003 \\
         --compare-label "E3/CTD-omitted anchor"] [--cutoff 8] [--window 250] \\
         [--reference refined_particles.star] [--metric z|pval] --out figures/correlogram_tile0003
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import mrcfile
import numpy as np

# RNF213 handoff palette (agent_context/PLOTS_AND_TEXT.md, tools/rnf213_style.py)
C = dict(full="#2878B5", partial="#A465B7", reference="#F0B43C", matched="#29B7E8",
         unmatched="#E55F72", text="#555555", score="#666666")
METRIC = {"z": ("search_mip_scaled.mrc", "Search z-score (flat-fielded SNR)"),
          "pval": ("search_pval.mrc", "−ln p (anisotropic Gaussian)"),
          "mip": ("search_mip.mrc", "Raw max CC")}


# ------------------------------------------------------------------ data

def read_map(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as m:
        a = np.asarray(m.data, dtype=np.float64)
    return a[0] if a.ndim == 3 else a          # [row, col] on the search grid


def par_params(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        f = line.split("#", 1)[0].split()
        if f:
            out[f[0]] = f[1:]
    return out


def tile_offset(tile: Path):
    off = tile.parent / "tile_offsets.csv"
    if off.exists():
        for r in csv.DictReader(off.open()):
            if r["tile"] == tile.name:
                return int(r["x0_zero_based"]), int(r["y0_zero_based"])
    return 0, 0


class Tile:
    def __init__(self, tile: Path, metric: str, label: str):
        self.tile, self.label = Path(tile), label
        out = self.tile / "output"
        name = METRIC[metric][0]
        if not (out / name).exists():
            raise SystemExit(f"{out / name} missing: run this tile with the Python port (write_mip 1)")
        self.map = read_map(out / name)
        self.z = read_map(out / "search_mip_scaled.mrc")
        self.pv = read_map(out / "search_pval.mrc") if (out / "search_pval.mrc").exists() else None
        p = par_params(self.tile / "search.par")
        self.apix = float(p["aPerPix"][0])
        self.ratio = float(p.get("aPerPix_search", p["aPerPix"])[0]) / self.apix
        img = self.tile / Path(p["imageFile"][0]).name
        with mrcfile.open(img, permissive=True, header_only=True) as m:
            self.img_shape = (int(m.header.ny), int(m.header.nx))
        self.cs = np.array(self.map.shape) // 2 + 1              # search-grid centre (1-based)
        self.co = np.array(self.img_shape) // 2 + 1              # tile-image centre (1-based)
        self.x0, self.y0 = tile_offset(self.tile)
        self.sources = {n: str(out / n) for n in (name, "search_mip_scaled.mrc")}
        self.cands = self._candidates(out)

    # map grid (1-based row/col, continuous) -> micrograph zero-based px
    def to_mic(self, r, c):
        y = (np.asarray(r, float) - self.cs[0]) * self.ratio + self.co[0] - 1 + self.y0
        x = (np.asarray(c, float) - self.cs[1]) * self.ratio + self.co[1] - 1 + self.x0
        return x, y

    def to_grid(self, x, y):
        r = (np.asarray(y, float) - self.y0 + 1 - self.co[0]) / self.ratio + self.cs[0]
        c = (np.asarray(x, float) - self.x0 + 1 - self.co[1]) / self.ratio + self.cs[1]
        return r, c

    def _candidates(self, out: Path):
        rows = []
        if (out / "particles.csv").exists():
            for r in csv.DictReader((out / "particles.csv").open()):
                rows.append(dict(id=int(r["num_id"]), score=float(r["score_final"]),
                                 r=float(r["row_search"]), c=float(r["col_search"])))
        elif (out / "list_final.txt").exists():
            lf = np.loadtxt(out / "list_final.txt", ndmin=2)
            pt = np.loadtxt(out / "particles.txt", ndmin=2) if (out / "particles.txt").exists() else None
            for i, v in enumerate(lf, 1):
                rows.append(dict(id=i, score=float(pt[i - 1, 1]) if pt is not None else float(v[0]),
                                 r=float(v[1]), c=float(v[2])))
        for d in rows:
            ri, ci = int(round(d["r"])) - 1, int(round(d["c"])) - 1
            d["z"] = float(self.z[ri, ci])
            d["neg_log_p"] = float(self.pv[ri, ci]) if self.pv is not None else float("nan")
            d["x"], d["y"] = (float(v) for v in self.to_mic(d["r"], d["c"]))
        return rows

    def window(self, x, y, half_A):
        """Sub-map centred on micrograph (x, y) px; returns arrays and Å coordinates."""
        r, c = self.to_grid(x, y)
        h = int(math.ceil(half_A / (self.apix * self.ratio)))
        r0, c0 = int(round(float(r))) - 1, int(round(float(c))) - 1
        rs = slice(max(0, r0 - h), min(self.map.shape[0], r0 + h + 1))
        cs = slice(max(0, c0 - h), min(self.map.shape[1], c0 + h + 1))
        sub = self.map[rs, cs]
        xs, _ = self.to_mic(np.full(sub.shape[1], rs.start + 1), np.arange(cs.start + 1, cs.stop + 1))
        _, ys = self.to_mic(np.arange(rs.start + 1, rs.stop + 1), np.full(sub.shape[0], cs.start + 1))
        return sub, xs * self.apix, ys * self.apix


def read_reference(path: Path, micrograph: str):
    """Reference picks of one micrograph (basename match), as in rnf213_workflow_v3.py."""
    labels, xy = [], []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("_rln"):
            labels.append(s.split()[0])
        elif s and not s.startswith(("data_", "loop_", "#", "_")) and labels:
            f = s.split()
            if len(f) == len(labels) and "_rlnCoordinateX" in labels:
                xy.append((float(f[labels.index("_rlnCoordinateX")]),
                           float(f[labels.index("_rlnCoordinateY")]),
                           f[labels.index("_rlnMicrographName")] if "_rlnMicrographName" in labels else ""))
    return [p for p in xy if Path(p[2]).name == micrograph]


# ------------------------------------------------------------------ drawing

def style():
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none"})


def save(fig, stem: Path, meta: dict):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("." + ext), dpi=300, bbox_inches="tight")
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=1))


def landscape(ax, sub, xs, ys, zlim, cutoff, cmap, title, floor=False, peak_label=True):
    import matplotlib as mpl
    from matplotlib.colors import LightSource, Normalize
    cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    X, Y = np.meshgrid(xs, ys)
    Z = np.clip(sub, zlim[0], zlim[1])
    norm = Normalize(zlim[0] - 0.3 * (zlim[1] - zlim[0]), zlim[1])   # floor drawn blue, not black
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(Z, cmap=cmap, norm=norm, vert_exag=0.15, blend_mode="soft")
    ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1, linewidth=0,
                    antialiased=False, shade=False)
    if floor:
        ax.contourf(X, Y, Z, zdir="z", offset=zlim[0], levels=24, cmap=cmap, norm=norm, alpha=0.55)
    if cutoff is not None and zlim[0] < cutoff < zlim[1]:
        ax.plot_surface(X[[0, -1]][:, [0, -1]], Y[[0, -1]][:, [0, -1]],
                        np.full((2, 2), cutoff), color="#9a9a9a", alpha=0.07, linewidth=0, shade=False)
        bx = [xs[0], xs[-1], xs[-1], xs[0], xs[0]]
        by = [ys[0], ys[0], ys[-1], ys[-1], ys[0]]
        ax.plot(bx, by, [cutoff] * 5, color=C["text"], lw=0.7, ls=(0, (2, 2)))
    if peak_label:
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        ax.text(xs[j], ys[i], Z[i, j] + 0.04 * (zlim[1] - zlim[0]), f"{sub[i, j]:.1f}",
                color="#222222", fontsize=7, ha="center")
    ax.set(xlim=(xs[0], xs[-1]), ylim=(ys[0], ys[-1]), zlim=zlim)
    ax.set_xlabel("x (Å)", labelpad=2)
    ax.set_ylabel("y (Å)", labelpad=2)
    ax.set_zlabel("z-score", labelpad=1)
    ax.view_init(elev=28, azim=-58)
    ax.set_box_aspect((1, (ys[-1] - ys[0]) / max(1e-9, xs[-1] - xs[0]), 0.55))
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor("white")
        a.pane.set_edgecolor("#dddddd")
        a._axinfo["grid"].update(color="#eeeeee", linewidth=0.5)
    ax.tick_params(labelsize=7, pad=0)
    ax.set_title(title, color="#333333", pad=0)


def topview(ax, sub, xs, ys, zlim, cutoff, cmap, peak=None):
    import matplotlib as mpl
    cm = (mpl.colormaps[cmap] if isinstance(cmap, str) else cmap).copy()
    cm.set_bad("#f2f2f2")                                    # excluded margin / padding
    sub = np.where(sub == 0, np.nan, sub)
    im = ax.imshow(sub, origin="upper", cmap=cm, vmin=zlim[0], vmax=zlim[1],
                   extent=(xs[0], xs[-1], ys[-1], ys[0]), interpolation="nearest")
    if cutoff is not None and np.nanmax(sub) >= cutoff:
        ax.contour(xs, ys, np.nan_to_num(sub), levels=[cutoff], colors="white", linewidths=0.9)
    if peak is not None:
        ax.plot(*peak, "+", color="white", ms=9, mew=1.2)
    ax.set(xlabel="x (Å)", ylabel="y (Å)")
    ax.spines[:].set_visible(False)
    return im


def profiles(ax, sub, xs, ys, cutoff, color):
    i, j = np.unravel_index(np.argmax(sub), sub.shape)
    ax.plot(xs - xs[j], sub[i, :], color=color, lw=1.6, label="along x")
    ax.plot(ys - ys[i], sub[:, j], color=color, lw=1.6, ls="--", label="along y")
    if cutoff is not None:
        ax.axhline(cutoff, color=C["text"], ls=":", lw=1.4)
        ax.text((xs - xs[j]).max(), cutoff, f"cutoff {cutoff:g}", color=C["text"],
                va="bottom", ha="right", fontsize=7)
    ax.set(xlabel="distance from peak (Å)", ylabel="z-score")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.16)


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", required=True)
    ap.add_argument("--label", default="Full RNF213")
    ap.add_argument("--compare", default=None, help="second tile folder (same micrograph region)")
    ap.add_argument("--compare-label", default="Partial template")
    ap.add_argument("--metric", choices=sorted(METRIC), default="z")
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--min-score", type=float, default=None, help="only candidates with final score >= this")
    ap.add_argument("--window", type=float, default=250.0, help="window edge in Å")
    ap.add_argument("--cmap", default="turbo", help="turbo (default), viridis, magma, jet ...")
    ap.add_argument("--zfloor", default="auto",
                    help="display floor for the landscapes: values below are drawn at the floor "
                         "(auto = ceil(median + 1 SD) of the map; 'none' shows everything)")
    ap.add_argument("--reference", default=None, help="RELION STAR with reference picks (markers only)")
    ap.add_argument("--micrograph", default=None,
                    help="micrograph name for the reference picks (default: the single *_fractions.mrc "
                         "in the current folder)")
    ap.add_argument("--max-candidates", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new --out (figures are never overwritten)")
    style()
    import matplotlib.pyplot as plt

    A = Tile(Path(a.tile), a.metric, a.label)
    B = Tile(Path(a.compare), a.metric, a.compare_label) if a.compare else None
    cands = [d for d in A.cands if a.min_score is None or d["score"] >= a.min_score]
    cands = sorted(cands, key=lambda d: -d["score"])[: a.max_candidates]
    cutoff = a.cutoff if a.metric == "z" else None
    zlab = METRIC[a.metric][1]
    maps = [A.map] + ([B.map] if B else [])
    vals = np.concatenate([m[m != 0] for m in maps])
    hi = math.ceil(float(vals.max()))
    if a.zfloor == "none":
        lo = math.floor(float(np.percentile(vals, 1)))
    elif a.zfloor == "auto":
        lo = math.ceil(float(np.median(vals) + vals.std()))
    else:
        lo = float(a.zfloor)
    zlim = (lo, max(hi, (cutoff or hi) + 1))                          # 3-D landscapes
    vlim = (math.floor(float(np.percentile(vals, 1))), zlim[1])         # top views, profiles
    refs = []
    if a.reference:
        mic = a.micrograph
        if mic is None:
            found = sorted(Path.cwd().glob("*_fractions.mrc"))
            if len(found) != 1:
                raise SystemExit("give --micrograph (no single *_fractions.mrc in the current folder)")
            mic = found[0].name
        refs = read_reference(Path(a.reference), Path(mic).name)
        print(f"{len(refs)} reference picks for {mic}")
    out.mkdir(parents=True)
    meta = dict(tile=str(A.tile), compare=str(B.tile) if B else None, metric=a.metric,
                metric_label=zlab, cutoff=cutoff, window_A=a.window, cmap=a.cmap,
                landscape_zlim=zlim, landscape_floor_note="values below the floor are drawn at "
                "the floor in the 3-D landscapes only (display choice)", topview_profile_range=vlim,
                note="search-stage map (best orientation per pixel, before refinement); final "
                     "scores are from optimisation and can differ",
                sources={k: v for t in ([A] + ([B] if B else [])) for k, v in
                         {f"{t.label}:{n}": p for n, p in t.sources.items()}.items()},
                sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                        for t in ([A] + ([B] if B else [])) for p in t.sources.values()})

    # ---- candidates.csv
    with open(out / "candidates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "final_score", "search_z", "neg_log_p", "x_px", "y_px", "x_A", "y_A",
                    "row_search", "col_search"])
        for d in A.cands:
            w.writerow([d["id"], d["score"], round(d["z"], 3), round(d["neg_log_p"], 3),
                        round(d["x"], 2), round(d["y"], 2), round(d["x"] * A.apix, 1),
                        round(d["y"] * A.apix, 1), d["r"], d["c"]])

    # ---- overview
    xs_all, _ = A.to_mic(np.ones(A.map.shape[1]), np.arange(1, A.map.shape[1] + 1))
    _, ys_all = A.to_mic(np.arange(1, A.map.shape[0] + 1), np.ones(A.map.shape[0]))
    fig, ax = plt.subplots(figsize=(6.4, 5.6), layout="constrained")
    # display only: 3x3 maximum filter so single-pixel peaks stay visible at full-tile scale
    from scipy.ndimage import maximum_filter
    shown = np.where(A.map == 0, 0, maximum_filter(A.map, size=3))
    im = topview(ax, shown, xs_all * A.apix, ys_all * A.apix, vlim, cutoff, a.cmap)
    for d in A.cands:
        col = "white" if a.min_score is None or d["score"] >= a.min_score else "#bbbbbb"
        ax.annotate(str(d["id"]), (d["x"] * A.apix, d["y"] * A.apix), xytext=(4, 4),
                    textcoords="offset points", color=col, fontsize=7, weight="bold")
        ax.plot(d["x"] * A.apix, d["y"] * A.apix, "o", mfc="none", mec=col, ms=7, mew=1.0)
    for x, y, _ in refs:
        X, Y = x * A.apix, y * A.apix
        if xs_all[0] * A.apix <= X <= xs_all[-1] * A.apix and ys_all[0] * A.apix <= Y <= ys_all[-1] * A.apix:
            ax.plot(X, Y, "o", mfc="none", mec=C["reference"], ms=11, mew=1.2)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label(zlab)
    cb.outline.set_visible(False)
    ax.set_title(f"RNF213 · {A.tile.parent.name}/{A.tile.name} · {A.label} · search-stage map"
                 + (f" · white contour = {cutoff:g}" if cutoff else ""), fontsize=9, color="#333333")
    ax.set_xlabel("x (Å)\n\nDisplay: 3×3 maximum filter · grey = excluded margin/padding · white = refined "
                  "candidates" + (" · amber = reference picks" if refs else ""), fontsize=8)
    save(fig, out / "overview", {**meta, "overview_display": "3x3 maximum filter (display only)"})
    plt.close(fig)

    # ---- per-candidate figures
    half = a.window / 2
    rows = [A] + ([B] if B else [])
    colors = [C["full"], C["partial"]]
    for d in cands:
        fig = plt.figure(figsize=(11, 3.6 * len(rows)), layout="constrained")
        gs = fig.add_gridspec(len(rows), 3, width_ratios=[1.6, 1.0, 1.0])
        for k, T in enumerate(rows):
            sub, xs, ys = T.window(d["x"], d["y"], half)
            ax3 = fig.add_subplot(gs[k, 0], projection="3d")
            landscape(ax3, sub, xs, ys, zlim, cutoff, a.cmap, T.label)
            ax2 = fig.add_subplot(gs[k, 1])
            topview(ax2, sub, xs, ys, vlim, cutoff, a.cmap, peak=(d["x"] * A.apix, d["y"] * A.apix))
            ax2.set_title(f"window max {sub.max():.2f}", fontsize=8, color=C["text"])
            ax1 = fig.add_subplot(gs[k, 2])
            profiles(ax1, sub, xs, ys, cutoff, colors[k])
            ax1.set_ylim(vlim)
        fig.suptitle(f"RNF213 · {A.tile.name} candidate {d['id']} · final SNR {d['score']:.2f} · "
                     f"x {d['x'] * A.apix:.0f} Å, y {d['y'] * A.apix:.0f} Å · {zlab}",
                     fontsize=10, color="#333333")
        save(fig, out / f"candidate_{d['id']:02d}", {**meta, "candidate": d})
        plt.close(fig)

    # ---- gallery
    n = len(cands)
    if n:
        ncol = min(4, n)
        nrow = math.ceil(n / ncol)
        fig = plt.figure(figsize=(3.1 * ncol, 2.7 * nrow), layout="constrained")
        for i, d in enumerate(cands):
            sub, xs, ys = A.window(d["x"], d["y"], half)
            ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
            landscape(ax, sub, xs, ys, zlim, cutoff, a.cmap,
                      f"#{d['id']} · final {d['score']:.2f}", floor=False)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")
            ax.tick_params(labelsize=5)
        fig.suptitle(f"RNF213 · {A.tile.name} · {A.label} · {zlab} landscapes ({a.window:g} Å windows, "
                     f"sorted by final SNR)", fontsize=10, color="#333333")
        save(fig, out / "gallery", meta)
        plt.close(fig)
    print(f"wrote {out}: overview, {len(cands)} candidate figures, gallery, candidates.csv")


if __name__ == "__main__":
    main()
