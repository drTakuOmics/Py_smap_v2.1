#!/usr/bin/env python
"""Orientation-free search for near-spherical particles (virus capsids, ferritin cages): positions,
pixel size, defocus and contrast sign estimated from the particles themselves (CPU, about a minute
per image).

A near-spherical particle projects to almost the same image in every orientation, so the rotational
average of its projected potential is one template for all particles. The image is whitened and
cross-correlated by FFT with that template, with a CTF applied; there is no orientation search. The
score of a setting is the mean z of the 10 best separate peaks; peaks above --thr are the particle
positions.

Input is either a pysmap search folder (search_status.json, searchImage.mrc and the template
potential; a short probe search is enough), or a micrograph with a 3D map of the particle and its CTF:

  python tools/radial_scan.py RESULTS/<micrograph>/ [...] --out DIAG
  python tools/radial_scan.py --image MIC.mrc --map PARTICLE.mrc --pixel 1.16 --defocus 15000 14600 35 --cs 2.7 --kv 300 --out DIAG

--map: a simulated potential (*_SP.mrc) or an EMDB map. --defocus: df1 df2 angle (A, A, degrees; the
ctffind, cisTEM and cryoSPARC convention) or one mean value.

Default: three scans, each with both contrast signs
  A  pixel size x0.5-2 of the given value, CTF at the given defocus
  B  defocus 0.2-5 um at the given pixel size
  C  pixel size with a 40 A low-pass and no CTF (independent of defocus)
--fine (calibration): pixel size x0.92-1.08 in 0.4% steps, with the defocus scaled by the squared pixel
ratio (what a CTF fit at the wrong pixel size gives), then defocus +-0.6 um at the best pixel size.

Outputs in --out: <name>_scan.png (scan curves; the image with a circle of the particle radius at each
peak above --thr), <name>_peaks.csv (positions in micrograph pixels, z) and, with --fine,
pixel_calibration.csv (all images calibrated into the same folder; "reliable" = at least --min-peaks
peaks and a best defocus inside the scanned range).

Reading the result:
  peaks on the particles at the given values -> pixel size and CTF are consistent with the image
  best pixel size, defocus or sign elsewhere  -> that value is off (read A and C first: B assumes
                                                 the given pixel size)
  normal contrast: with CTF (A, B) "as built" wins; without CTF (C) "flipped" wins (dark particles)
Positions and calibration only, no orientations; z is not a 2DTM score. Strong edges (carbon, ice)
also produce peaks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # repo root (smap) when run as a script

S_COARSE = np.r_[0.5, 0.6, 0.7, 0.8, np.arange(0.86, 1.15, 0.02), 1.2, 1.3, 1.5, 1.75, 2.0]
DF_COARSE = np.arange(2000, 50001, 1000.0)                        # A
SIGNS = ((1, "as built"), (-1, "flipped"))
CAL_FIELDS = ["micrograph", "pixel_A", "df1_A", "df2_A", "angle_deg", "given_pixel_A", "given_df1_A",
              "given_df2_A", "contrast", "top10_z", "top10_z_given", "peaks", "reliable"]


def wavelength_A(V: float) -> float:
    """Electron wavelength (A) at V volts, as used by the search (smap.ctf)."""
    from smap.ctf import wavelength
    return wavelength(V) * 1e10


def par_value(text: str, key: str):
    m = re.search(rf"^{key}\s+(.+)$", text, re.M)
    return m.group(1).split() if m else None


def particle_profile(vol: np.ndarray, vox: float) -> tuple:
    """Rotational average of the projection of a 3D map: radii (A), profile, particle radius (A)."""
    v = np.asarray(vol, dtype=np.float64)
    v = v - np.median(v)
    p = v.sum(axis=2)
    c = (np.array(p.shape) - 1) / 2
    yy, xx = np.indices(p.shape)
    rb = np.round(np.hypot(yy - c[0], xx - c[1])).astype(int)
    prof = np.bincount(rb.ravel(), p.ravel()) / np.maximum(np.bincount(rb.ravel()), 1)
    prof = prof[: min(p.shape) // 2 + 1]                           # inside the inscribed circle
    r = np.arange(prof.size) * vox
    R = float(r[np.nonzero(prof > 0.05 * prof.max())[0].max()])
    return r, prof, R


class Scanner:
    """Whitened, binned image and the radial template; score() evaluates one setting."""

    def __init__(self, im, pix, profile, Cs_A, V, ang=0.0, amp=0.07, bin_=2, thr=8.0, sep_A=None, lowpass_A=None):
        r_A, prof, R = profile
        a = np.asarray(im, dtype=np.float64)
        nz = np.argwhere(a != 0)                                   # drop zero padding
        (r0, c0), (r1, c1) = nz.min(0), nz.max(0) + 1
        a = a[r0:r1, c0:c1]
        b = int(bin_)
        n0, n1 = (a.shape[0] // b) * b, (a.shape[1] // b) * b
        a = np.clip((a[:n0, :n1] - a.mean()) / a.std(), -8, 8)         # hot pixels
        if b > 1:                         # band-limit to the new Nyquist, then decimate: no aliasing
            F = np.fft.rfft2(a)
            F[(np.abs(np.fft.fftfreq(n0))[:, None] > 0.5 / b) | (np.fft.rfftfreq(n1)[None, :] > 0.5 / b)] = 0
            a = np.fft.irfft2(F, s=(n0, n1))[::b, ::b]
        a = (a - a.mean()) / a.std()
        self.img, self.crop, self.b, self.pix0 = a, (int(r0), int(c0)), b, pix * b
        self.r_A, self.prof, self.R = r_A, prof, R
        self.Cs, self.lam, self.ang, self.amp = Cs_A, wavelength_A(V), ang, amp
        self.thr, self.sep_A, self.lowpass_A = thr, sep_A or 1.5 * R, lowpass_A
        F = np.fft.rfft2(a)
        fy = np.fft.fftfreq(a.shape[0])[:, None]
        fx = np.fft.rfftfreq(a.shape[1])[None, :]
        self.kpx = np.hypot(fy, fx)                                # cycles per binned pixel
        self.alpha = np.arctan2(-fy, fx)                           # as smap.ctf: x = column, y = -row
        kb = np.minimum((self.kpx * min(a.shape)).astype(int), min(a.shape) // 2)
        psd = np.bincount(kb.ravel(), (np.abs(F) ** 2).ravel()) / np.maximum(np.bincount(kb.ravel()), 1)
        self.wsd = np.sqrt(ndimage.uniform_filter1d(psd, 3)[kb] + 1e-12)
        self.Fw = F / self.wsd
        yy = np.fft.fftfreq(a.shape[0]) * a.shape[0]
        xx = np.fft.fftfreq(a.shape[1]) * a.shape[1]
        self.rpx = np.hypot(yy[:, None], xx[None, :])              # binned pixels from the origin
        e = int(np.ceil(1.2 * R / self.pix0))
        if 2 * e + 10 > min(a.shape):
            raise ValueError(f"image ({a.shape[0]} x {a.shape[1]} binned pixels) too small for a {R:.0f} A particle")
        self.inner = np.zeros(a.shape, bool)
        self.inner[e:-e, e:-e] = True

    def ctf(self, k, df1, df2):
        df = 0.5 * (df1 + df2 + (df1 - df2) * np.cos(2 * (self.alpha - self.ang)))
        chi = np.pi * self.lam * df * k ** 2 - 0.5 * np.pi * self.Cs * self.lam ** 3 * k ** 4
        return -(math.sqrt(1 - self.amp ** 2) * np.sin(chi) + self.amp * np.cos(chi))

    def score(self, s, df1, df2, sign=1, use_ctf=True, lp_A=None, keep=False):
        """Mean z of the 10 best separate peaks and the number above thr, if the true pixel size is s x
        the given one. keep=True also returns the peak positions (binned pixels) and the z map."""
        pix = self.pix0 * s
        k = self.kpx / pix
        T = np.fft.rfft2(np.interp(self.rpx * pix, self.r_A, self.prof, right=0.0))
        if use_ctf:
            T = T * self.ctf(k, df1, df2)
        lp = lp_A or self.lowpass_A
        band = (k > 1 / 400.0) & (k < min(1 / lp if lp else np.inf, 1 / (2.05 * pix)))
        cc = np.fft.irfft2(self.Fw * np.conj(sign * T * band / self.wsd), s=self.img.shape)
        v = cc[self.inner]
        med = np.median(v)
        z = (cc - med) / (1.4826 * np.median(np.abs(v - med)))
        mx = ndimage.maximum_filter(np.where(self.inner, z, -np.inf), size=max(3, int(round(self.sep_A / pix))))
        loc = (z == mx) & self.inner
        top = float(np.sort(z[loc])[::-1][:10].mean()) if loc.any() else float("nan")
        pk = loc & (z > self.thr)
        if keep:
            pos = np.argwhere(pk)
            return top, len(pos), pos[np.argsort(-z[pk])], z
        return top, int(pk.sum())


def coarse(sc: Scanner, df1: float, df2: float) -> dict:
    h = (df1 - df2) / 2
    res = {}
    for sign, _ in SIGNS:
        res["A", sign] = [sc.score(s, df1, df2, sign) for s in S_COARSE]
        res["B", sign] = [sc.score(1.0, d + h, d - h, sign) for d in DF_COARSE]
        res["C", sign] = [sc.score(s, 0, 0, sign, use_ctf=False, lp_A=40.0) for s in S_COARSE]
    return res


def fine(sc: Scanner, df1: float, df2: float, sign: int, lo=0.92, hi=1.08, step=0.004) -> dict:
    S = np.arange(lo, hi + step / 2, step)
    m_s = np.array([sc.score(s, df1 * s * s, df2 * s * s, sign)[0] for s in S])
    i = int(np.nanargmax(m_s))
    s_b = float(S[i])
    if 0 < i < len(S) - 1:                                         # parabola through the top 3
        y0, y1, y2 = m_s[i - 1:i + 2]
        den = y0 - 2 * y1 + y2
        if den < 0:
            s_b += 0.5 * (y0 - y2) / den * step
    off = np.arange(-6000, 6001, 500.0)
    m_d = np.array([sc.score(s_b, df1 * s_b ** 2 + o, df2 * s_b ** 2 + o, sign)[0] for o in off])
    return dict(S=S, m_s=m_s, s=s_b, off=off, m_d=m_d, d_off=float(off[int(np.nanargmax(m_d))]))


def read_map(path: Path, vox=None, fallback=None):
    import mrcfile
    from smap import io as sio
    path = Path(path)
    if path.suffix.lower() in (".cif", ".mmcif", ".pdb", ".ent"):
        raise SystemExit(f"{path}: --map needs a 3D map (.mrc), not an atomic model; use the *_SP.mrc "
                         "written by a search, or an EMDB map")
    if not path.exists():
        raise SystemExit(f"{path}: not found (prefixes mapped with --path-map in a search are not "
                         "resolved here)")
    with mrcfile.open(path, permissive=True, header_only=True) as m:
        v = vox or float(m.voxel_size.x) or fallback
    if not v:
        raise SystemExit(f"{path}: no voxel size in the header; pass --map-pixel")
    vol = sio.mr(path)
    if vol.ndim != 3:
        raise SystemExit(f"{path}: not a 3D map")
    return vol, float(v)


def from_folder(f: Path, map_override=None) -> dict:
    from smap import io as sio
    if not (f / "search_status.json").exists():
        raise SystemExit(f"{f}: no search_status.json (not a finished pysmap search folder)")
    st = json.loads((f / "search_status.json").read_text())
    txt = st.get("provenance", {}).get("params_text", "")
    if not txt:
        raise SystemExit(f"{f}: search_status.json has no params_text")
    a_orig = float(par_value(txt, "aPerPix")[0])
    a_srch = float((par_value(txt, "aPerPix_search") or [a_orig])[0])
    d = [float(x) for x in (par_value(txt, "defocus") or ["0"])[:3]]
    if len(d) < 3:                                                 # mean defocus only
        d = [d[0], d[-1] if len(d) == 2 else d[0], 0.0]
    if (par_value(txt, "defocus_format") or [""])[0] == "CTFFind":
        df1, df2, ang = d[0], d[1], -math.radians(d[2])            # A, A, deg -> smap angle
    else:
        df1, df2, ang = 10 * d[0], 10 * d[1], d[2]                 # nm, nm, rad
    sp = Path(map_override) if map_override else next(iter(sorted(f.glob("*_SP.mrc"))), None)
    if sp is None:
        sf = (par_value(txt, "structureFile") or [""])[0]
        sp = Path(sf) if sf.endswith(".mrc") and Path(sf).exists() else None
    if sp is None:
        raise SystemExit(f"{f}: no template potential (*_SP.mrc in the folder or as structureFile); "
                         "pass --map")
    return dict(name=f.name, image=sio.mr(f / "searchImage.mrc"), pix=a_srch, pix_given=a_orig,
                df1=df1, df2=df2, ang=ang, Cs_A=float((par_value(txt, "Cs") or ["2.7e-3"])[0]) * 1e10,
                V=float((par_value(txt, "V_acc") or ["300000"])[0]), map=sp, map_fallback=a_orig,
                mic_scale=a_srch / a_orig, mic_offset=False)


def from_args(a) -> dict:
    from smap import io as sio
    d = list(a.defocus)
    if len(d) == 1:
        d = [d[0], d[0], 0.0]
    if len(d) != 3:
        raise SystemExit("--defocus takes one value (mean, A) or three (df1 A, df2 A, angle deg)")
    return dict(name=a.name or Path(a.image).stem, image=sio.mr(a.image), pix=a.pixel, pix_given=a.pixel,
                df1=d[0], df2=d[1], ang=-math.radians(d[2]), Cs_A=a.cs * 1e7, V=a.kv * 1e3, map=Path(a.map),
                map_fallback=None, mic_scale=1.0, mic_offset=True)


def write_peaks(path: Path, sc: Scanner, pos, z, it: dict, s, df1, df2, sign) -> None:
    b, (r0, c0) = sc.b, sc.crop
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x_px", "y_px", "z", "radius_px", "pixel_A", "df1_A", "df2_A", "contrast"])
        for (y, x) in pos:
            yc, xc = y * b, x * b
            if it["mic_offset"]:
                yc, xc = yc + r0, xc + c0
            w.writerow([f"{xc * it['mic_scale']:.1f}", f"{yc * it['mic_scale']:.1f}", f"{z[y, x]:.2f}",
                        f"{sc.R / (s * it['pix_given']):.1f}", f"{s * it['pix_given']:.4f}", f"{df1:.0f}",
                        f"{df2:.0f}", SIGNS[0][1] if sign == 1 else SIGNS[1][1]])


def show_image(ax, sc: Scanner, pos, title: str, plt, s: float = 1.0) -> None:
    disp = ndimage.gaussian_filter(sc.img, 2)
    lo, hi = np.percentile(disp, [1, 99])
    ax.imshow(disp, cmap="gray", vmin=lo, vmax=hi)
    for (y, x) in pos:
        ax.add_patch(plt.Circle((x, y), sc.R / (sc.pix0 * s), fill=False, color="#1baf7a", lw=0.8))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folders", nargs="*", help="pysmap search folders")
    ap.add_argument("--image", help="micrograph (.mrc), instead of a search folder")
    ap.add_argument("--map", help="3D map of the particle (default for folders: the search's *_SP.mrc)")
    ap.add_argument("--map-pixel", type=float, default=None, help="voxel size of --map if not in its header (A)")
    ap.add_argument("--pixel", type=float, help="pixel size of --image (A)")
    ap.add_argument("--defocus", type=float, nargs="+", help="df1 df2 angle (A, A, deg) or mean defocus (A)")
    ap.add_argument("--cs", type=float, default=2.7, help="spherical aberration (mm), with --image")
    ap.add_argument("--kv", type=float, default=300.0, help="acceleration voltage (kV), with --image")
    ap.add_argument("--amp", type=float, default=0.07, help="amplitude contrast")
    ap.add_argument("--name", default=None, help="output name for --image (default: file name)")
    ap.add_argument("--out", default="radial_scan")
    ap.add_argument("--bin", type=int, default=None, help="binning of the image (default: about 3 A per pixel)")
    ap.add_argument("--thr", type=float, default=8.0, help="z of a reported peak")
    ap.add_argument("--sep-A", type=float, default=None, help="minimum peak distance (default 1.5 x particle radius)")
    ap.add_argument("--lowpass", type=float, default=None,
                    help="resolution limit of the CTF template (A; default Nyquist); 15-20 A can find more "
                         "particles in weak images, at the cost of pixel-size precision")
    ap.add_argument("--fine", action="store_true", help="calibration of pixel size and defocus (see above)")
    ap.add_argument("--fine-range", type=float, nargs=2, default=(0.92, 1.08), metavar=("LO", "HI"))
    ap.add_argument("--min-peaks", type=int, default=5,
                    help="--fine: images with fewer peaks above --thr are marked not reliable")
    a = ap.parse_args(argv)
    if a.image:
        if a.pixel is None or a.defocus is None or a.map is None:
            ap.error("--image needs --map, --pixel and --defocus")
        items = [lambda: from_args(a)]
    elif a.folders:
        items = [lambda f=Path(f): from_folder(f, a.map) for f in a.folders]
    else:
        ap.error("give search folders or --image")
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for load in items:
        it = load()
        vol, vox = read_map(it["map"], a.map_pixel, it["map_fallback"])
        profile = particle_profile(vol, vox)
        del vol
        b = a.bin or max(1, int(round(3.0 / it["pix"])))
        sc = Scanner(it["image"], it["pix"], profile, it["Cs_A"], it["V"], it["ang"], a.amp, b, a.thr, a.sep_A,
                     a.lowpass)
        name, pg, df1, df2 = it["name"], it["pix_given"], it["df1"], it["df2"]
        dfm = (df1 + df2) / 2
        print(f"\n{name}: pixel {pg:g} A, defocus {df1:.0f} / {df2:.0f} A, angle {-math.degrees(it['ang']):.1f} deg, "
              f"Cs {it['Cs_A'] / 1e7:.3g} mm, {it['V'] / 1e3:.0f} kV; particle radius {sc.R:.0f} A; "
              f"{sc.img.shape[1]} x {sc.img.shape[0]} pixels of {sc.pix0:.2f} A", flush=True)

        if a.fine:
            c_up, c_dn = (sc.score(1.0, 0, 0, g, use_ctf=False, lp_A=40.0)[0] for g, _ in SIGNS)
            sign = 1 if c_dn >= c_up else -1                       # dark particles <-> CTF "as built"
            r = fine(sc, df1, df2, sign, *a.fine_range)
            s = r["s"]
            d1, d2 = df1 * s * s + r["d_off"], df2 * s * s + r["d_off"]
            top, n, pos, z = sc.score(s, d1, d2, sign, keep=True)
            top_u, n_u = sc.score(1.0, df1, df2, sign)
            lab = SIGNS[0][1] if sign == 1 else SIGNS[1][1]
            weak, edge = n < a.min_peaks, abs(r["d_off"]) >= r["off"].max()
            note = (f"; fewer than {a.min_peaks} peaks: not reliable" if weak else "") + \
                   ("; best defocus at the edge of the scan" if edge else "")
            print(f"  calibrated: pixel {s * pg:.4f} A, defocus {(d1 + d2) / 2 / 1e4:.3f} um (given {dfm / 1e4:.3f}, "
                  f"scaled {dfm * s * s / 1e4:.3f}), contrast {lab}: top-10 z {top:.1f}, {n} peaks > {a.thr:g}; "
                  f"at the given values {top_u:.1f}, {n_u} peaks{note}", flush=True)
            row = dict(micrograph=name, pixel_A=f"{s * pg:.4f}", df1_A=f"{d1:.0f}", df2_A=f"{d2:.0f}",
                       angle_deg=f"{-math.degrees(it['ang']):.2f}", given_pixel_A=f"{pg:.4f}", given_df1_A=f"{df1:.0f}",
                       given_df2_A=f"{df2:.0f}", contrast=lab, top10_z=f"{top:.2f}", top10_z_given=f"{top_u:.2f}",
                       peaks=n, reliable="no" if weak or edge else "yes")
            with open(out / f"{name}.calibration.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=CAL_FIELDS)
                w.writeheader()
                w.writerow(row)
            fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
            ax[0].plot(r["S"] * pg, r["m_s"], "o-", ms=3, color="#2a78d6")
            ax[0].axvline(pg, color="#6b6b6b", ls="--", lw=0.8)
            ax[0].set_xlabel("pixel size (A), defocus scaled by pixel ratio squared")
            ax[1].plot((dfm * s * s + r["off"]) / 1e4, r["m_d"], "o-", ms=3, color="#2a78d6")
            ax[1].axvline(dfm / 1e4, color="#6b6b6b", ls="--", lw=0.8)
            ax[1].set_xlabel(f"mean defocus (um) at pixel {s * pg:.4f} A")
            for x_ in ax[:2]:
                x_.set_ylabel("mean z of the 10 best peaks")
            show_image(ax[2], sc, pos, f"{n} peaks > {a.thr:g} at the calibrated values ({lab})", plt, s)
            write_peaks(out / f"{name}_peaks.csv", sc, pos, z, it, s, d1, d2, sign)
        else:
            res = coarse(sc, df1, df2)
            names = {"A": "pixel size, CTF at given defocus", "B": "defocus, given pixel size",
                     "C": "pixel size, no CTF (40 A)"}
            grids = {"A": (S_COARSE * pg, "A/px"), "B": (DF_COARSE / 1e4, "um"), "C": (S_COARSE * pg, "A/px")}
            for scan in "ABC":
                g_, unit = grids[scan]
                for sign, lab in SIGNS:
                    m = np.array([x[0] for x in res[scan, sign]])
                    i = int(np.nanargmax(m))
                    print(f"  {scan} {names[scan]:33s} {lab:8s}: best {g_[i]:6.3f} {unit}  (top-10 mean z "
                          f"{m[i]:5.1f}, {res[scan, sign][i][1]:3d} peaks > {a.thr:g})")
            for sign, lab in SIGNS:
                t_u, n_u = sc.score(1.0, df1, df2, sign)
                print(f"  given values {lab:8s}: top-10 mean z {t_u:5.1f}, {n_u:3d} peaks > {a.thr:g}")
            print("  normal contrast: A, B 'as built' > 'flipped'; C 'flipped' > 'as built'")
            sign = max((1, -1), key=lambda g: np.nanmax([x[0] for x in res["B", g]]))
            dB = DF_COARSE[int(np.nanargmax([x[0] for x in res["B", sign]]))]
            h = (df1 - df2) / 2
            top, n, pos, z = sc.score(1.0, dB + h, dB - h, sign, keep=True)
            fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
            labels = {"A": "pixel size (A), CTF at given defocus", "B": "defocus (um), given pixel size",
                      "C": "pixel size (A), no CTF, 40 A low-pass"}
            for k_, scan in enumerate("ABC"):
                for sg, col, lab in ((1, "#2a78d6", "as built"), (-1, "#eb6834", "flipped")):
                    ax[k_].plot(grids[scan][0], [x[0] for x in res[scan, sg]], "o-", ms=3, color=col, label=lab)
                ax[k_].axvline(pg if scan != "B" else dfm / 1e4, color="#6b6b6b", ls="--", lw=0.8)
                ax[k_].set_xlabel(labels[scan])
                ax[k_].set_ylabel("mean z of the 10 best peaks")
                ax[k_].set_title(scan, loc="left")
                ax[k_].legend(frameon=False, fontsize=8)
            lab = SIGNS[0][1] if sign == 1 else SIGNS[1][1]
            show_image(ax[3], sc, pos, f"{n} peaks > {a.thr:g} at defocus {dB / 1e4:.1f} um ({lab})", plt)
            write_peaks(out / f"{name}_peaks.csv", sc, pos, z, it, 1.0, dB + h, dB - h, sign)
        fig.suptitle(name, fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"{name}_scan.png", dpi=100, facecolor="white")
        plt.close(fig)
        print(f"  -> {out / (name + '_scan.png')}, {out / (name + '_peaks.csv')}", flush=True)

    if a.fine:
        rows = []
        for p in sorted(out.glob("*.calibration.csv")):              # one row per image
            rows += list(csv.DictReader(open(p)))
        with open(out / "pixel_calibration.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CAL_FIELDS)
            w.writeheader()
            w.writerows(rows)
        for lab, sel in (("all", rows), ("reliable", [r for r in rows if r.get("reliable") == "yes"])):
            px = np.array([float(r["pixel_A"]) for r in sel])
            if len(px):
                print(f"pixel size, {lab} images in {out} ({len(px)}): median {np.median(px):.4f} A "
                      f"(range {px.min():.4f}-{px.max():.4f})")
        print(f"-> {out / 'pixel_calibration.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
