#!/usr/bin/env python
"""Detections of several searches of the same micrograph in one figure, one colour per search
(for example two states of a complex), drawn as found: every detection with final score >= its
cutoff is rendered at its pose, overlaps included.

    python tools/overlay_figures.py CLOSED_RESULT OPEN_RESULT --labels "TRiC closed" "TRiC open"
        --colors "#17a2a2" "#d63fa0" --cutoff 7.8 --out FIG_DIR [--same-A 75]

Surfaces as in tools/search_figures.py (template potential of each search, rotated to each pose,
seen along the beam). The first search is drawn as filled surfaces; the others as outlines in their
colour over a light fill, so that a particle found by two searches shows both. Detections of different searches closer than --same-A
(default: 0.3 x the larger template diameter) are counted as found at the same position; nothing is
assigned or removed.

Writes (never overwritten) FIG_DIR/overlay.png (whole micrograph), overlay_pairs.png (top positions:
micrograph | each search alone | overlay), overlay.csv (every detection: search, id, score, position, nearest
detection of each other search and its distance and score) and overlay.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # repo root (smap)
sys.path.insert(0, str(Path(__file__).resolve().parent))         # tools/

DEFAULT_COLORS = ("#17a2a2", "#d63fa0", "#eda100", "#2a78d6")    # teal, magenta, amber, blue


def outlined(rgba: np.ndarray, color, width: int, fill_alpha: float) -> np.ndarray:
    """Sprite with a full-opacity outline of ``width`` pixels in ``color`` and a light interior."""
    from matplotlib.colors import to_rgb
    from scipy.ndimage import binary_erosion
    hit = rgba[..., 3] > 0
    out = rgba.copy()
    out[..., 3] = np.where(hit, fill_alpha, 0.0)
    edge = hit & ~binary_erosion(hit, iterations=max(1, width))
    out[edge, :3] = to_rgb(color)
    out[edge, 3] = 1.0
    return out


def load(res: Path, cutoff: float):
    """Detections of one search at its cutoff, with the pixel size, the micrograph record and the template volume."""
    from search_figures import find_volume, par_value, read_particles, status
    st = status(res)
    text = st.get("provenance", {}).get("params_text", "")
    a_orig = float(par_value(text, "aPerPix") or 0) or None
    rec = st.get("provenance", {}).get("input_sha256", {}).get("imageFile", {}) or {}
    img, sha = rec.get("path", ""), rec.get("sha256") or ""
    if a_orig is None:
        raise SystemExit(f"{res}: search_status.json lacks aPerPix")
    parts = read_particles(res)
    for d in parts:
        d["xA"], d["yA"] = d["x"] * a_orig, d["y"] * a_orig
    det = sorted([d for d in parts if d["score"] >= cutoff], key=lambda d: -d["score"])
    return dict(res=res, a=a_orig, img=img, sha=sha, vol=find_volume(res, st), det=det, n_refined=len(parts))


def main(argv=None) -> int:
    from search_figures import (bin_mean, draw_poses, prepare_volume, render, rotate_volume, save, scale_bar,
                                show_micrograph, style, voxel_size)
    from smap import io as sio

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", nargs="+", help="search output folders of the same micrograph")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--colors", nargs="+", default=None)
    ap.add_argument("--cutoff", type=float, nargs="+", required=True, help="one value, or one per search")
    ap.add_argument("--out", required=True)
    ap.add_argument("--micrograph", default=None)
    ap.add_argument("--same-A", type=float, default=None, help="distance counted as the same position")
    ap.add_argument("--fill-alpha", type=float, default=0.15, help="interior opacity of the outlined searches")
    ap.add_argument("--lighting", default="soft", choices=("soft", "simple", "flat"))
    ap.add_argument("--render-A", type=float, default=1.5)
    ap.add_argument("--surface-sigma-A", type=float, default=1.5)
    ap.add_argument("--overview-render-A", type=float, default=3.0)
    ap.add_argument("--overview-sigma-A", type=float, default=3.0)
    ap.add_argument("--mass-frac", type=float, default=0.85)
    ap.add_argument("--display-bin-A", type=float, default=4.0)
    ap.add_argument("--n-pairs", type=int, default=4)
    ap.add_argument("--pair-half-A", type=float, default=450.0)
    a = ap.parse_args(argv)
    n = len(a.results)
    labels = a.labels or [Path(r).parent.name for r in a.results]
    colors = a.colors or list(DEFAULT_COLORS[:n])
    cuts = a.cutoff * n if len(a.cutoff) == 1 else a.cutoff
    if not (len(labels) == len(colors) == len(cuts) == n):
        raise SystemExit("give one label, colour and cutoff per search (or one cutoff for all)")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new --out (figures are never overwritten)")
    S = [load(Path(r), c) for r, c in zip(a.results, cuts)]
    ids = {s["sha"] or s["img"] for s in S}                     # same file content (or path) for every search
    if a.micrograph is None and len(ids) > 1:
        raise SystemExit("the searches are of different micrographs: " + ", ".join(sorted({s["img"] for s in S})))
    if max(abs(s["a"] - S[0]["a"]) for s in S) > 1e-4:
        raise SystemExit("the searches have different pixel sizes")
    img_path = Path(a.micrograph or S[0]["img"])
    if not img_path.is_file():
        raise SystemExit(f"micrograph {img_path} not found; pass --micrograph")
    a_orig = S[0]["a"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.ndimage import gaussian_filter
    style()

    img = sio.mr(img_path).astype(np.float64)
    b = max(1, int(round(a.display_bin_A / a_orig)))
    disp = gaussian_filter(bin_mean(img, b), 1.0)
    ext = (0, disp.shape[1] * b * a_orig, disp.shape[0] * b * a_orig, 0)
    b2 = max(1, int(round(2.0 / a_orig)))
    disp2 = gaussian_filter(bin_mean(img, b2), 2.0)
    ext2 = (0, disp2.shape[1] * b2 * a_orig, disp2.shape[0] * b2 * a_orig, 0)
    del img

    diam = 0.0
    for k, s in enumerate(S):
        vol = sio.mr(s["vol"])
        av = voxel_size(s["vol"], a_orig)
        V, lev, vox = prepare_volume(vol, av, a.render_A, a.mass_frac, a.surface_sigma_A)
        Vo, levo, voxo = prepare_volume(vol, av, a.overview_render_A, a.mass_frac, a.overview_sigma_A)
        del vol
        s["box"], s["box_o"] = V.shape[0] * vox, Vo.shape[0] * voxo
        diam = max(diam, s["box"] - 6 * vox)
        s["sprites"] = {d["id"]: render(rotate_volume(V, d["q"]), lev, colors[k], a.lighting, 0.95, vox)
                        for d in s["det"]}                                   # filled (each search alone)
        s["sprites_o"] = {d["id"]: render(rotate_volume(Vo, d["q"]), levo, colors[k], a.lighting, 0.95, voxo)
                          for d in s["det"]}
        if k:                                                                # overlay: outline + light fill
            s["ov"] = {i: outlined(x, colors[k], 2, a.fill_alpha) for i, x in s["sprites"].items()}
            s["ov_o"] = {i: outlined(x, colors[k], 2, a.fill_alpha) for i, x in s["sprites_o"].items()}
        else:
            s["ov"], s["ov_o"] = s["sprites"], s["sprites_o"]
        print(f"{labels[k]}: {len(s['det'])} of {s['n_refined']} refined candidates >= {cuts[k]:g}", flush=True)
    same = a.same_A or 0.3 * diam

    # nearest detection of every other search (reported, nothing assigned)
    table = []
    for k, s in enumerate(S):
        for d in s["det"]:
            row = dict(search=labels[k], id=d["id"], score=round(d["score"], 3), x_A=round(d["xA"], 1),
                       y_A=round(d["yA"], 1))
            for j, t in enumerate(S):
                if j == k:
                    continue
                if t["det"]:
                    dist = [np.hypot(d["xA"] - e["xA"], d["yA"] - e["yA"]) for e in t["det"]]
                    i = int(np.argmin(dist))
                    row[f"nearest_{labels[j]}_A"] = round(float(dist[i]), 1)
                    row[f"nearest_{labels[j]}_score"] = round(t["det"][i]["score"], 3)
                else:
                    row[f"nearest_{labels[j]}_A"], row[f"nearest_{labels[j]}_score"] = "", ""
            table.append(row)
    both = 0
    if n >= 2 and S[0]["det"] and S[1]["det"]:
        both = sum(1 for d in S[0]["det"] if min(np.hypot(d["xA"] - e["xA"], d["yA"] - e["yA"]) for e in S[1]["det"]) < same)

    out.mkdir(parents=True)
    legend = [Patch(facecolor=colors[k], edgecolor="#333333", label=f"{labels[k]}: {len(s['det'])} (≥ {cuts[k]:g})")
              for k, s in enumerate(S)]
    note = f"{both} positions found by both {labels[0]} and {labels[1]} (< {same:.0f} Å)" if n >= 2 else ""

    fig, ax = plt.subplots(figsize=(8.6, 8.6 * disp.shape[0] / disp.shape[1]))
    show_micrograph(ax, disp, ext)
    for s in S:
        draw_poses(ax, s["ov_o"], s["det"], s["box_o"])
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    scale_bar(ax, 1000.0, ext)
    ax.legend(handles=legend, loc="upper left", fontsize=8, framealpha=0.85)
    ax.set_title(f"{img_path.stem}: detections as found" + (f"; {note}" if note else ""), fontsize=9)
    save(fig, out / "overlay.png")
    plt.close(fig)

    # top positions: the best detections of all searches, one window per position
    cand = sorted(((d["score"], k, d) for k, s in enumerate(S) for d in s["det"]), key=lambda t: -t[0])
    wins = []
    for _, k, d in cand:
        if all(np.hypot(d["xA"] - w["xA"], d["yA"] - w["yA"]) > a.pair_half_A for w in wins):
            wins.append(d)
        if len(wins) >= a.n_pairs:
            break
    if wins:
        ncol = n + 2
        fig, axs = plt.subplots(len(wins), ncol, figsize=(3.3 * ncol, 3.4 * len(wins)), squeeze=False)
        h = a.pair_half_A
        for r, d in enumerate(wins):
            cx = min(max(d["xA"], ext[0] + h), ext[1] - h)
            cy = min(max(d["yA"], ext[3] + h), ext[2] - h)
            for j in range(ncol):
                axx = axs[r, j]
                show_micrograph(axx, disp2, ext2)
                if 1 <= j <= n:                                              # one search alone, filled
                    s = S[j - 1]
                    draw_poses(axx, s["sprites"], s["det"], s["box"])
                elif j == n + 1:                                             # all searches
                    for s in S:
                        draw_poses(axx, s["ov"], s["det"], s["box"])
                axx.set_xlim(cx - h, cx + h)
                axx.set_ylim(cy + h, cy - h)
                axx.set(xticks=[], yticks=[], xlabel="", ylabel="")
            here = []
            for k, s in enumerate(S):
                near = [e for e in s["det"] if np.hypot(e["xA"] - d["xA"], e["yA"] - d["yA"]) < same]
                here.append(f"{labels[k]} {max(e['score'] for e in near):.1f}" if near else f"{labels[k]} —")
            axs[r, 0].set_ylabel("\n".join(here), fontsize=8)
        axs[0, 0].set_title("micrograph")
        for k in range(n):
            axs[0, k + 1].set_title(labels[k], color=colors[k])
        axs[0, n + 1].set_title("overlay")
        fig.legend(handles=legend, loc="lower center", ncol=n, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        save(fig, out / "overlay_pairs.png")
        plt.close(fig)

    keys = list(dict.fromkeys(k for r in table for k in r))
    with open(out / "overlay.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(table)
    meta = dict(micrograph=str(img_path), searches=[dict(result=str(s["res"]), label=labels[k], color=colors[k],
                                                         cutoff=cuts[k], volume=str(s["vol"]), detected=len(s["det"]),
                                                         refined=s["n_refined"]) for k, s in enumerate(S)],
                same_position_A=round(same, 1), found_by_first_two=both)
    (out / "overlay.json").write_text(json.dumps(meta, indent=1))
    print(f"-> {out}: overlay.png, overlay_pairs.png, overlay.csv" + (f"; {note}" if note else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
