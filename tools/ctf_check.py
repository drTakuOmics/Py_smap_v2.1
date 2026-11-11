#!/usr/bin/env python
"""Check a search .par's defocus against the Thon rings of its micrograph, in the port's layout.

The micrograph power spectrum (averaged over tiles, original sampling) is compared with the
port's own CTF² (smap.ctf, the function the search uses) within a resolution band, after the
same high-pass on both (removes the smooth background). Scores (Pearson r):

- ``given``: the .par values;
- ``angle_flipped``: astigmatism angle with the opposite sign (convention error);
- ``df_swapped``: df1 and df2 exchanged (angle off by 90 deg);
- a scan of the mean defocus (``--scan-A``), whose best offset should be ~0.

    python tools/ctf_check.py pars/<micrograph>.par [--tile 512] [--band 30 4] [--png ctf_check.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # repo root (smap) when run as a script

import argparse
import json
import math

import numpy as np


def tile_psd(im: np.ndarray, n: int) -> np.ndarray:
    """Mean |FFT|² of non-overlapping n×n tiles (mean removed, edge taper), centred."""
    im = np.asarray(im, dtype=np.float64)
    r = np.hanning(n)
    w = np.outer(r, r)
    acc = np.zeros((n, n))
    k = 0
    for i in range(0, im.shape[0] - n + 1, n):
        for j in range(0, im.shape[1] - n + 1, n):
            t = im[i:i + n, j:j + n]
            t = (t - t.mean()) * w
            acc += np.abs(np.fft.fft2(t)) ** 2
            k += 1
    if k == 0:
        raise ValueError(f"image {im.shape} smaller than the tile {n}")
    return np.fft.fftshift(acc / k)


def model_ctf2(n: int, params: dict, df_row) -> np.ndarray:
    """CTF squared (n, n) from the port's own CTF function for one defocus row [df1 nm, df2 nm, angle rad]."""
    from smap.ctf import ctf
    return np.imag(ctf(np.atleast_2d(df_row), [n, n], params)[:, :, 0]) ** 2


def _highpass(x, sigma):
    """x minus its Gaussian-smoothed copy (removes the smooth background)."""
    from scipy.ndimage import gaussian_filter
    return x - gaussian_filter(x, sigma)


def score(A_hp, M, band, sigma) -> float:
    """Pearson correlation, inside the resolution band, of the high-passed spectrum with the high-passed model."""
    m = _highpass(M, sigma)[band]
    a = A_hp[band]
    m, a = m - m.mean(), a - a.mean()
    return float((a * m).sum() / math.sqrt((a * a).sum() * (m * m).sum()))


def check(params: dict, im: np.ndarray, tile: int = 512, band_A=(30.0, 4.0), sigma: float = 4.0,
          scan_A: float = 1000.0, step_A: float = 50.0):
    """Scores of the .par defocus against the Thon rings: as given, with the astigmatism angle
    flipped, with df1 and df2 swapped, and the best common defocus offset within +/- scan_A.
    Returns (scores, high-passed amplitude spectrum, band mask, model CTF squared)."""
    from smap.utils import get_ks
    a = float(params["aPerPix"])
    df = np.atleast_1d(np.asarray(params["defocus"], dtype=np.float64)).copy()[:3]
    if params.get("defocus_format") == "CTFFind":          # as in search_global.prepare
        df[0:2] = df[0:2] / 10
        df[2] = df[2] * (-math.pi / 180)
    P = tile_psd(im, tile)
    k, _ = get_ks(tile, a)
    lo, hi = 1 / band_A[0], 1 / max(band_A[1], 2.05 * a)
    band = (k > lo) & (k < hi)
    A_hp = _highpass(np.sqrt(P), sigma)
    p = dict(params, aPerPix=a)
    out = {"given": score(A_hp, model_ctf2(tile, p, df), band, sigma),
           "angle_flipped": score(A_hp, model_ctf2(tile, p, [df[0], df[1], -df[2]]), band, sigma),
           "df_swapped": score(A_hp, model_ctf2(tile, p, [df[1], df[0], df[2]]), band, sigma)}
    offs = np.arange(-scan_A, scan_A + step_A / 2, step_A)
    sc = [score(A_hp, model_ctf2(tile, p, [df[0] + o / 10, df[1] + o / 10, df[2]]), band, sigma)
          for o in offs]
    i = int(np.argmax(sc))
    out.update(best_offset_A=float(offs[i]), best_offset_score=float(sc[i]),
               band_A=[band_A[0], round(1 / hi, 2)], tile=tile,
               defocus_nm_rad=[float(x) for x in df])
    return out, A_hp, band, model_ctf2(tile, p, df)


def main(argv=None) -> int:
    from smap import io as sio
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("par")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--band", type=float, nargs=2, default=(30.0, 4.0), metavar=("LOW_A", "HIGH_A"))
    ap.add_argument("--scan-A", type=float, default=1000.0)
    ap.add_argument("--png", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    params, _ = sio.read_params_file(a.par)
    im = sio.mr(params["imageFile"])
    res, A_hp, band, M = check(params, im, a.tile, tuple(a.band), scan_A=a.scan_A)
    res["par"] = str(a.par)
    print(f"{Path(a.par).name}: r given {res['given']:.3f} | angle flipped {res['angle_flipped']:.3f} "
          f"| df1/df2 swapped {res['df_swapped']:.3f} | best mean-defocus offset "
          f"{res['best_offset_A']:+.0f} A (r {res['best_offset_score']:.3f})")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter
        n = a.tile
        d = np.where(band, A_hp, 0)
        m = np.where(band, M - gaussian_filter(M, 4.0), 0)
        img = np.zeros((n, n))
        img[:, : n // 2] = d[:, : n // 2] / (d[band].std() or 1)
        img[:, n // 2:] = m[:, n // 2:] / (m[band].std() or 1)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img, cmap="gray", vmin=-2.5, vmax=2.5)
        ax.set_title(f"left: micrograph   right: model (r = {res['given']:.2f})", fontsize=9)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(a.png, dpi=150)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
