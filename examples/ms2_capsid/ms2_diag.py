#!/usr/bin/env python
"""Why no MS2 capsid reached the cutoff: contrast sign vs pixelwise flat-fielding, from the maps
the search already wrote (CPU only, no new search).

For a near-spherical capsid every orientation correlates at low resolution, so the
orientation-mean map (search_mean.mrc) is itself a capsid detector:
  mean map has strong POSITIVE discs  -> capsids are there with the right contrast; the z-score
                                         (mip - mean) / SD removes that signal (flat-fielding)
  mean map has strong NEGATIVE discs  -> image contrast is inverted relative to the template
  neither                             -> no capsid signal at all (pixel size, CTF, wrong images)

  cd $PYSMAP
  python examples/ms2_capsid/ms2_diag.py $MS2_WORK/results_a3.8/*/ --out $MS2_WORK/diag
Prints one line per micrograph; writes diag/<micrograph>.png (image | mean | raw MIP | z).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def robust(a: np.ndarray) -> tuple:
    v = a[np.isfinite(a)]
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med)) or 1.0
    return med, mad


def lowpass(a: np.ndarray, sigma_px: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(a, sigma_px)


def binned(a: np.ndarray, b: int) -> np.ndarray:
    n0, n1 = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    return a[:n0, :n1].reshape(n0 // b, b, n1 // b, b).mean((1, 3))


def main(argv=None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (smap)
    from smap import io as sio
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", default="ms2_diag")
    ap.add_argument("--edge-px", type=int, default=100, help="ignore this many search pixels at the border")
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"{'micrograph':40s} {'mean+':>6s} {'mean-':>6s} {'skew':>6s} {'MIPmax':>7s} {'z@MIP':>6s} "
          f"{'mean@MIP':>8s} {'SD@MIP/med':>10s} {'zmax':>5s}")
    for f in map(Path, a.folders):
        need = ["search_mean.mrc", "search_SD.mrc", "search_mip.mrc", "search_mip_scaled.mrc"]
        if not all((f / n).exists() for n in need):
            print(f"{f.name[:40]:40s} missing maps ({', '.join(n for n in need if not (f / n).exists())})")
            continue
        mean3 = sio.mr(f / "search_mean.mrc").astype(np.float64)
        sd3 = sio.mr(f / "search_SD.mrc").astype(np.float64)
        mip = sio.mr(f / "search_mip.mrc").astype(np.float64)
        z = sio.mr(f / "search_mip_scaled.mrc").astype(np.float64)
        best = sio.mr(f / "search_mip_dfInds.mrc").astype(int) - 1 if (f / "search_mip_dfInds.mrc").exists() else None
        if mean3.ndim == 2:
            mean3, sd3 = mean3[:, :, None], sd3[:, :, None]
        k = mean3.shape[2] // 2
        e = a.edge_px
        inner = np.zeros(mip.shape, bool)
        inner[e:-e, e:-e] = True
        inner &= mip != 0
        mean = mean3[:, :, k]
        mmed, mmad = robust(mean[inner])
        mz = (mean - mmed) / mmad                                  # mean map in robust-SD units
        v = mz[inner]
        skew = float(np.mean(((v - v.mean()) / v.std()) ** 3))
        i = np.unravel_index(np.argmax(np.where(inner, mip, -np.inf)), mip.shape)
        pk = best[i] if best is not None else k
        sd_med = np.median(sd3[:, :, pk][inner])
        print(f"{f.name[:40]:40s} {np.max(v):6.1f} {np.min(v):6.1f} {skew:6.2f} {mip[i]:7.2f} {z[i]:6.2f} "
              f"{mz[i]:8.1f} {sd3[i[0], i[1], pk] / sd_med:10.2f} {np.max(z[inner]):5.2f}")

        img = sio.mr(f / "searchImage.mrc").astype(np.float64) if (f / "searchImage.mrc").exists() else None
        b = 4
        fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
        panels = []
        if img is not None:
            il = binned(lowpass(img, 4.0), b)
            lo, hi = np.percentile(il, [1, 99])
            panels.append(("image (low-pass ~15 Å)", il, "gray", lo, hi))
        else:
            panels.append(("image (searchImage.mrc missing)", np.zeros((2, 2)), "gray", 0, 1))
        mzb = binned(np.where(inner, mz, 0), b)
        lim = max(6.0, np.percentile(np.abs(mzb), 99.5))
        panels.append(("orientation-mean CC (robust SD)\nred = positive, blue = negative", mzb, "RdBu_r", -lim, lim))
        mb = binned(np.where(inner, mip, np.median(mip[inner])), b)
        lo, hi = np.percentile(mb, [1, 99.9])
        panels.append(("raw MIP (max CC over orientations)", mb, "magma", lo, hi))
        zb = binned(np.where(inner, z, 0), b)
        panels.append(("z = (MIP - mean) / SD  (what is thresholded)", zb, "viridis", 0, max(7.0, zb.max())))
        for axx, (t, im, cm, lo, hi) in zip(ax, panels):
            h = axx.imshow(im, cmap=cm, vmin=lo, vmax=hi, origin="upper")
            axx.set_title(t, fontsize=9)
            axx.set_xticks([])
            axx.set_yticks([])
            plt.colorbar(h, ax=axx, shrink=0.75, pad=0.02)
        fig.suptitle(f"{f.name}   mean map extremes +{np.max(v):.1f} / {np.min(v):.1f} robust SD, skew {skew:.2f}",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"{f.name}.png", dpi=110, facecolor="white")
        plt.close(fig)
    print(f"-> {out}/*.png")
    print("mean+ / mean-: largest positive / negative value of the orientation-mean map (robust SD units)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
