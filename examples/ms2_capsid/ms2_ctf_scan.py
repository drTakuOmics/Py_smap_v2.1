#!/usr/bin/env python
"""Is the MS2 failure the pixel size, the contrast or the defocus, or inside the search? (CPU, ~1 min each)

Uses the image exactly as the search used it (searchImage.mrc) and a capsid template that does not
depend on orientation: the rotational average of the 2MS2 potential's projection (a capsid is nearly
round, so one template fits every virion). The image is whitened and cross-correlated with that
template under a CTF; the score is the mean z of the 10 best separate peaks (and the number of
peaks above z = 8), scanned over:
  A  pixel size (x 0.5 - 2 of the value used), CTF at the cryoSPARC defocus
  B  defocus 0.2 - 5 um at the pixel size used (which defocus finds the virions best?)
  C  pixel size again with a low-pass template and no CTF (independent of defocus)
Both contrast signs are tried. Normal images (protein dark): with CTF (A, B) the template "as built"
wins; without CTF (C) the "flipped" one wins (the dark virion disc). The opposite = inverted image.
  peaks on the virions at the values the search used -> image, pixel size and CTF are fine; the
      problem is inside the 2DTM search
  peaks only at another pixel size / defocus / sign   -> that parameter is wrong

  cd $PYSMAP
  python examples/ms2_capsid/ms2_ctf_scan.py $MS2_WORK/results_a3.8/FoilHole_19046868*/ --out $MS2_WORK/diag
Calibration on all micrographs (pixel size and defocus, one line each + diag/ms2_pixel_calibration.csv):
  python examples/ms2_capsid/ms2_ctf_scan.py $MS2_WORK/results_a3.8/*/ --fine --out $MS2_WORK/diag
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

def wavelength(V: float) -> float:
    return 12.2643 / np.sqrt(V * (1 + 0.978466e-6 * V))


def par_value(text: str, key: str):
    m = re.search(rf"^{key}\s+(.+)$", text, re.M)
    return m.group(1).split() if m else None


def radial_profile(img2d: np.ndarray, step: float) -> tuple:
    c = (np.array(img2d.shape) - 1) / 2
    yy, xx = np.indices(img2d.shape)
    r = np.hypot(yy - c[0], xx - c[1])
    rb = np.round(r).astype(int)
    prof = np.bincount(rb.ravel(), img2d.ravel()) / np.maximum(np.bincount(rb.ravel()), 1)
    return np.arange(prof.size) * step, prof


def kgrid(shape, pix):
    ky = np.fft.fftfreq(shape[0], d=pix)
    kx = np.fft.fftfreq(shape[1], d=pix)
    return np.hypot(ky[:, None], kx[None, :])


def ctf(k, df_A, Cs_A, lam, amp=0.07):
    chi = np.pi * lam * df_A * k ** 2 - 0.5 * np.pi * Cs_A * lam ** 3 * k ** 4
    return -(np.sqrt(1 - amp ** 2) * np.sin(chi) + amp * np.cos(chi))


def count_peaks(z, inner, pix, thr=8.0, sep_A=200.0):
    """Separate local maxima (>= sep_A apart): number above thr, positions, mean of the 10 best."""
    size = max(3, int(round(sep_A / pix)))
    mx = ndimage.maximum_filter(np.where(inner, z, -np.inf), size=size)
    pk = (z == mx) & (z > thr) & inner
    loc = (z == mx) & inner
    top = np.sort(z[loc])[::-1][:10].mean() if loc.any() else np.nan
    return int(pk.sum()), np.argwhere(pk), top


def main(argv=None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (smap)
    from smap import io as sio
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", default="ms2_diag")
    ap.add_argument("--bin", type=int, default=2, help="bin the search image (1.5 A) by this")
    ap.add_argument("--fine", action="store_true",
                    help="calibration only: pixel x0.92-1.02 in 0.4%% steps (defocus scaled by pixel^2, as a CTF "
                         "fit at the wrong pixel would be), then defocus +-0.6 um at the best pixel; one line each")
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = []
    for f in map(Path, a.folders):
        st = json.loads((f / "search_status.json").read_text())
        txt = st.get("provenance", {}).get("params_text", "")
        a_orig = float(par_value(txt, "aPerPix")[0])
        a_srch = float((par_value(txt, "aPerPix_search") or [a_orig])[0])
        d = [float(x) for x in par_value(txt, "defocus")[:3]]
        df_used = (d[0] + d[1]) / 2                                  # CTFFind format, A
        Cs_A = float(par_value(txt, "Cs")[0]) * 1e10
        V = float((par_value(txt, "V_acc") or [300000])[0])
        lam = wavelength(V)
        sps = sorted(f.glob("*_SP.mrc"))
        if not sps:
            print(f"{f.name}: no *_SP.mrc")
            continue
        import mrcfile
        with mrcfile.open(sps[0], permissive=True, header_only=True) as m:
            vox = float(m.voxel_size.x) or a_orig
        SP = sio.mr(sps[0]).astype(np.float64)
        SP -= np.median(SP)
        r_A, prof = radial_profile(SP.sum(2), vox)
        R_cap = r_A[np.nonzero(prof > 0.05 * prof.max())[0].max()]

        im = sio.mr(f / "searchImage.mrc").astype(np.float64)
        nz = np.argwhere(im != 0)
        (r0, c0), (r1, c1) = nz.min(0), nz.max(0) + 1
        im = im[r0:r1, c0:c1]
        b = a.bin
        n0, n1 = (im.shape[0] // b) * b, (im.shape[1] // b) * b
        im = im[:n0, :n1].reshape(n0 // b, b, n1 // b, b).mean((1, 3))
        im = (im - im.mean()) / im.std()
        F = np.fft.fft2(im)
        P = np.abs(F) ** 2
        kpx = kgrid(im.shape, 1.0)
        kb = np.minimum((kpx * min(im.shape)).astype(int), min(im.shape) // 2)
        psd = np.bincount(kb.ravel(), P.ravel()) / np.maximum(np.bincount(kb.ravel()), 1)
        psd = ndimage.uniform_filter1d(psd, 3)[kb]
        Fw = F / np.sqrt(psd + 1e-12)
        edge = int(np.ceil(1.2 * R_cap / (a_srch * b)))
        inner = np.zeros(im.shape, bool)
        inner[edge:-edge, edge:-edge] = True

        def score(s, df_A, sign, use_ctf=True, lp_A=None, keep=False):
            pix = a_srch * b * s                           # true A per binned pixel
            k = kpx / pix
            yy = np.fft.fftfreq(im.shape[0]) * im.shape[0]
            xx = np.fft.fftfreq(im.shape[1]) * im.shape[1]
            rad = np.hypot(yy[:, None], xx[None, :]) * pix
            t = np.interp(rad, r_A, prof, right=0.0)
            T = np.fft.fft2(t)
            if use_ctf:
                T = T * ctf(k, df_A, Cs_A, lam)
            band = (k > 1 / 400.0) & (k < (1 / lp_A if lp_A else 1 / (2.05 * pix)))
            Tw = sign * T * band / np.sqrt(psd + 1e-12)
            cc = np.real(np.fft.ifft2(Fw * np.conj(Tw)))
            v = cc[inner]
            med = np.median(v)
            z = (cc - med) / (1.4826 * np.median(np.abs(v - med)))
            n, pos, top = count_peaks(z, inner, pix)
            return (n, pos, top, z) if keep else (n, top)

        if a.fine:
            S_f = np.arange(0.92, 1.0201, 0.004)
            m_s = np.array([score(s, df_used * s * s, 1)[1] for s in S_f])
            i = int(np.nanargmax(m_s))
            s_b = S_f[i]
            if 0 < i < len(S_f) - 1:                               # parabola through the top 3
                y0, y1, y2 = m_s[i - 1:i + 2]
                den = y0 - 2 * y1 + y2
                s_b = S_f[i] + (0.5 * (y0 - y2) / den * 0.004 if den < 0 else 0.0)
            df0 = df_used * s_b * s_b
            DFf = df0 + np.arange(-6000, 6001, 500.0)
            m_d = np.array([score(s_b, d_, 1)[1] for d_ in DFf])
            j = int(np.nanargmax(m_d))
            n_b, m_b = score(s_b, DFf[j], 1)
            n_u, m_u = score(1.0, df_used, 1)
            summary.append((f.name, s_b * a_orig, DFf[j], df_used, m_b))
            print(f"{f.name[:40]:40s} pixel {s_b * a_orig:.4f} A  defocus {DFf[j] / 1e4:.2f} um "
                  f"(cryoSPARC {df_used / 1e4:.2f}, scaled {df0 / 1e4:.2f})  top-10 z {m_b:4.1f} "
                  f"({n_b} > 8)  vs used {m_u:4.1f} ({n_u} > 8)", flush=True)
            continue
        res = {}
        S = np.r_[0.5, 0.6, 0.7, 0.8, np.arange(0.86, 1.15, 0.02), 1.2, 1.3, 1.5, 1.75, 2.0]
        DF = np.arange(2000, 50001, 1000.0)
        for sign in (1, -1):
            res[("A", sign)] = [score(s, df_used, sign) for s in S]
            res[("B", sign)] = [score(1.0, df, sign) for df in DF]
            res[("C", sign)] = [score(s, 0, sign, use_ctf=False, lp_A=40.0) for s in S]

        def best(key, grid):
            m = np.array([x[1] for x in res[key]])
            i = int(np.nanargmax(m))
            return grid[i], m[i], res[key][i][0]
        print(f"\n{f.name}  (search used: pixel {a_orig} A, defocus {df_used / 1e4:.2f} um, Cs {Cs_A / 1e7:.3g} mm; "
              f"capsid radius in template {R_cap:.0f} A)")
        lines = []
        names = {"A": "pixel size, CTF at search defocus", "B": "defocus, search pixel size",
                 "C": "pixel size, no CTF (40 A)"}
        for scan, grid, unit in (("A", S * a_orig, "A/px"), ("B", DF / 1e4, "um"), ("C", S * a_orig, "A/px")):
            for s_, lab in ((1, "as built"), (-1, "flipped")):
                g, m, n = best((scan, s_), grid)
                print(f"  {scan} {names[scan]:34s} {lab:8s}: best {g:6.3f} {unit}  "
                      f"(top-10 mean z {m:5.1f}, {n:3d} peaks z>8)")
        for s_, lab in ((1, "as built"), (-1, "flipped")):
            n_u, m_u = score(1.0, df_used, s_)
            print(f"  values the search used (pixel {a_orig} A, defocus {df_used / 1e4:.2f} um) {lab:8s}: "
                  f"top-10 mean z {m_u:5.1f}, {n_u:3d} peaks z>8")
        print("  normal contrast: A/B 'as built' >> 'flipped', C 'flipped' >> 'as built'; noise level is z ~ 5")

        # figure: curves + image with peaks at the best scan-B setting
        fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
        for k_, (scan, grid, xl) in enumerate((("A", S * a_orig, "pixel size (A), CTF at search defocus"),
                                               ("B", DF / 1e4, "defocus (um), search pixel size"),
                                               ("C", S * a_orig, "pixel size (A), no CTF, 40 A low-pass"))):
            for sign, col, lab in ((1, "#2a78d6", "as built"), (-1, "#eb6834", "flipped")):
                ax[k_].plot(grid, [x[1] for x in res[(scan, sign)]], "o-", ms=3, color=col, label=lab)
            ax[k_].axvline(a_orig if scan != "B" else df_used / 1e4, color="#6b6b6b", ls="--", lw=0.8)
            ax[k_].set_xlabel(xl)
            ax[k_].set_ylabel("mean z of the 10 best peaks")
            ax[k_].set_title(f"{scan}", loc="left")
            ax[k_].legend(frameon=False, fontsize=8)
        sB = max((1, -1), key=lambda s: np.nanmax([x[1] for x in res[("B", s)]]))
        dfB = DF[int(np.nanargmax([x[1] for x in res[("B", sB)]]))]
        n, pos, top, z = score(1.0, dfB, sB, keep=True)
        disp = ndimage.gaussian_filter(im, 2)
        lo, hi = np.percentile(disp, [1, 99])
        ax[3].imshow(disp, cmap="gray", vmin=lo, vmax=hi)
        rpx = R_cap / (a_srch * b)
        for (y, x) in pos:
            ax[3].add_patch(plt.Circle((x, y), rpx, fill=False, color="#1baf7a", lw=0.8))
        ax[3].set_title(f"peaks at defocus {dfB / 1e4:.1f} um ({'as built' if sB == 1 else 'flipped'}): {n}",
                        fontsize=9)
        ax[3].set_xticks([])
        ax[3].set_yticks([])
        fig.suptitle(f.name, fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"{f.name}_scan.png", dpi=100, facecolor="white")
        plt.close(fig)
        print(f"  -> {out / (f.name + '_scan.png')}")
    if summary:
        px = np.array([s[1] for s in summary])
        print(f"\npixel size from {len(px)} micrographs: median {np.median(px):.4f} A "
              f"(range {px.min():.4f}-{px.max():.4f}); search used {a_orig}")
        with open(out / "ms2_pixel_calibration.csv", "w") as fh:
            fh.write("micrograph,pixel_A,defocus_A,cryosparc_defocus_A,top10_z\n")
            for s in summary:
                fh.write(f"{s[0]},{s[1]:.4f},{s[2]:.0f},{s[3]:.0f},{s[4]:.2f}\n")
        print(f"-> {out / 'ms2_pixel_calibration.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
