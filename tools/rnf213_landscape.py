#!/usr/bin/env python3
"""One 3-D z-score landscape for a whole micrograph: all port-searched tiles stitched, every
detected candidate labelled (the gallery's peaks in one shared space).

The per-pixel z-score maps (search_mip_scaled.mrc) of every tile listed in
<tree>/tile_offsets.csv that was searched with the port are placed in micrograph
coordinates (same conversion as rnf213_workflow_v3.py) and max-binned onto a common grid
(--bin Å; overlapping tiles take the maximum). Max-binning keeps every peak height and
raises the displayed noise floor slightly (max over a bin); it is a display choice and is
recorded in the JSON sidecar. Tiles not (yet) searched with the port are drawn grey.

Candidates: refined candidates of every tile (particles.csv), merged across overlapping
tiles within --dedup-radius px (highest final score kept, as in the review scripts),
labelled "id: final SNR" when final SNR >= --cutoff (others drawn as small grey ticks).

Outputs (never overwritten): landscape.{png,pdf,svg,json} (3-D landscape + top view),
landscape_candidates.csv.

usage (project root, port environment):
  python ../Py_smap_v2.1/tools/rnf213_landscape.py --tree tiles_3p8_port --out figures/landscape_3p8
  python ../Py_smap_v2.1/tools/rnf213_landscape.py --tree batch_noE3_30_port/searches/image_0001 \\
         --label "E3/CTD-omitted anchor" --out figures/landscape_anchor_img0001
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("rnf213_correlogram", Path(__file__).with_name("rnf213_correlogram.py"))
corr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(corr)


def stitch(tree: Path, bin_A: float):
    rows = list(csv.DictReader((tree / "tile_offsets.csv").open()))
    tiles, missing = [], []
    for r in rows:
        t = tree / r["tile"]
        if (t / "output" / "search_mip_scaled.mrc").exists():
            tiles.append((r, corr.Tile(t, "z", r["tile"])))
        else:
            missing.append(r)
    if not tiles:
        raise SystemExit(f"no tile in {tree} has search_mip_scaled.mrc (run the port first)")
    apix = tiles[0][1].apix
    W = max(int(r["x0_zero_based"]) + int(r["width"]) for r in rows)
    H = max(int(r["y0_zero_based"]) + int(r["height"]) for r in rows)
    nx, ny = math.ceil(W * apix / bin_A), math.ceil(H * apix / bin_A)
    grid = np.full((ny, nx), -np.inf)
    for _, T in tiles:
        rr, cc = np.nonzero(T.z)
        x, y = T.to_mic(rr + 1, cc + 1)
        ix = np.clip((x * apix / bin_A).astype(int), 0, nx - 1)
        iy = np.clip((y * apix / bin_A).astype(int), 0, ny - 1)
        np.maximum.at(grid, (iy, ix), T.z[rr, cc])
    grid[~np.isfinite(grid)] = np.nan
    xs = (np.arange(nx) + 0.5) * bin_A
    ys = (np.arange(ny) + 0.5) * bin_A
    return grid, xs, ys, tiles, missing, apix


def merge_candidates(tiles, radius_px):
    allc = []
    for r, T in tiles:
        for d in T.cands:
            allc.append(dict(d, tile=r["tile"]))
    kept = []
    for d in sorted(allc, key=lambda d: -d["score"]):
        if all(math.hypot(d["x"] - k["x"], d["y"] - k["y"]) > radius_px for k in kept):
            kept.append(d)
    for i, d in enumerate(kept, 1):
        d["label_id"] = i
    return kept


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True, help="folder with tile_offsets.csv (one micrograph)")
    ap.add_argument("--label", default="Full RNF213")
    ap.add_argument("--bin", type=float, default=12.0, help="display bin in Å (max-binning)")
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--dedup-radius", type=float, default=20.0, help="px, as in the review scripts")
    ap.add_argument("--zfloor", type=float, default=None, help="landscape floor (default: median + 1 SD)")
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--elev", type=float, default=32)
    ap.add_argument("--azim", type=float, default=-62)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new --out (figures are never overwritten)")
    corr.style()
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource, Normalize

    tree = Path(a.tree)
    grid, xs, ys, tiles, missing, apix = stitch(tree, a.bin)
    cands = merge_candidates(tiles, a.dedup_radius)
    vals = grid[np.isfinite(grid)]
    floor = a.zfloor if a.zfloor is not None else math.ceil(float(np.median(vals) + vals.std()))
    top = max(math.ceil(float(vals.max())), a.cutoff + 1)
    zlim = (floor, top)
    cm = mpl.colormaps[a.cmap]
    Z = np.where(np.isfinite(grid), np.clip(grid, *zlim), zlim[0])
    norm = Normalize(zlim[0] - 0.3 * (zlim[1] - zlim[0]), zlim[1])
    rgb = LightSource(azdeg=315, altdeg=45).shade(Z, cmap=cm, norm=norm, vert_exag=0.05, blend_mode="soft")
    rgb[~np.isfinite(grid)] = (0.92, 0.92, 0.92, 1.0)               # not searched with the port

    out.mkdir(parents=True)
    fig = plt.figure(figsize=(15, 6.6), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0])
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    X, Y = np.meshgrid(xs, ys)
    ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    if zlim[0] < a.cutoff < zlim[1]:
        ax.plot_surface(X[[0, -1]][:, [0, -1]], Y[[0, -1]][:, [0, -1]], np.full((2, 2), a.cutoff),
                        color="#9a9a9a", alpha=0.06, linewidth=0, shade=False)
    for d in cands:
        X0, Y0 = d["x"] * apix, d["y"] * apix
        ix = min(len(xs) - 1, int(X0 / a.bin))
        iy = min(len(ys) - 1, int(Y0 / a.bin))
        zpk = Z[iy, ix]
        if d["score"] >= a.cutoff:
            ax.text(X0, Y0, zpk + 0.05 * (zlim[1] - zlim[0]), f"{d['label_id']}: {d['score']:.1f}",
                    fontsize=6.5, color="#222222", ha="center", zorder=10)
    ax.set(xlim=(xs[0], xs[-1]), ylim=(ys[-1], ys[0]), zlim=zlim)    # y down, as in the micrograph
    ax.set_xlabel("x (Å)", labelpad=4)
    ax.set_ylabel("y (Å)", labelpad=4)
    ax.set_zlabel("z-score", labelpad=2)
    ax.view_init(elev=a.elev, azim=a.azim)
    ax.set_box_aspect((xs[-1] - xs[0], ys[-1] - ys[0], 0.35 * max(xs[-1] - xs[0], ys[-1] - ys[0])))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor("#dddddd")
        axis._axinfo["grid"].update(color="#eeeeee", linewidth=0.5)
    ax.tick_params(labelsize=7, pad=0)

    ax2 = fig.add_subplot(gs[0, 1])
    cm2 = cm.copy()
    cm2.set_bad("#ececec")
    vlo = math.floor(float(np.percentile(vals, 1)))
    im = ax2.imshow(grid, cmap=cm2, vmin=vlo, vmax=zlim[1], origin="upper",
                    extent=(0, xs[-1] + a.bin / 2, ys[-1] + a.bin / 2, 0), interpolation="nearest")
    for d in cands:
        X0, Y0 = d["x"] * apix, d["y"] * apix
        ok = d["score"] >= a.cutoff
        ax2.plot(X0, Y0, "o", mfc="none", mec="white" if ok else "#bbbbbb", ms=6 if ok else 4, mew=0.9)
        if ok:
            ax2.annotate(str(d["label_id"]), (X0, Y0), xytext=(3, 3), textcoords="offset points",
                         color="white", fontsize=6.5, weight="bold")
    for r in [r for r, _ in tiles] + missing:
        x0, y0 = int(r["x0_zero_based"]) * apix, int(r["y0_zero_based"]) * apix
        ax2.add_patch(plt.Rectangle((x0, y0), int(r["width"]) * apix, int(r["height"]) * apix, fill=False,
                                    ec="#ffdb6e" if r in missing else "#83dfde", lw=0.4, alpha=0.7))
    ax2.set(xlabel="x (Å)", ylabel="y (Å)")
    ax2.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax2, shrink=0.75, pad=0.01)
    cb.set_label("Search z-score (flat-fielded SNR)")
    cb.outline.set_visible(False)
    n8 = sum(d["score"] >= a.cutoff for d in cands)
    fig.suptitle(f"RNF213 · {tree.as_posix()} · {a.label} · {len(tiles)}/{len(tiles) + len(missing)} sections "
                 f"searched with the port · {n8} merged candidates with final SNR ≥ {a.cutoff:g} "
                 f"(labels id: final SNR)", fontsize=10, color="#333333")
    fig.text(0.01, 0.005, f"Display: {a.bin:g} Å max-binning; landscape floor {zlim[0]:g}; grey = not searched "
             "with the port; section outlines cyan (searched) / amber (not searched). Heights are search-stage "
             "z-scores (before refinement); labels give final optimised SNR.", fontsize=7.5, color="#555555")
    meta = dict(tree=str(tree), label=a.label, bin_A=a.bin, zlim=zlim, cutoff=a.cutoff,
                dedup_radius_px=a.dedup_radius, tiles_used=[r["tile"] for r, _ in tiles],
                tiles_missing=[r["tile"] for r in missing],
                note="max-binned display of search_mip_scaled.mrc; overlapping sections take the maximum")
    corr.save(fig, out / "landscape", meta)
    plt.close(fig)
    with open(out / "landscape_candidates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label_id", "tile", "tile_candidate_id", "final_score", "search_z", "x_px", "y_px",
                    "x_A", "y_A"])
        for d in cands:
            w.writerow([d["label_id"], d["tile"], d["id"], d["score"], round(d["z"], 3), round(d["x"], 2),
                        round(d["y"], 2), round(d["x"] * apix, 1), round(d["y"] * apix, 1)])
    print(f"wrote {out}: landscape ({len(tiles)} sections, {len(cands)} merged candidates, {n8} >= {a.cutoff:g})")


if __name__ == "__main__":
    main()
