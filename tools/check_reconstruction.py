#!/usr/bin/env python
"""Reconstruction from a RELION particle STAR file, compared with the template: a check of the
particle images and orientations before cryoSPARC or RELION.

    python tools/check_reconstruction.py EXPORT/particles.star --template SP.mrc --out EXPORT/check
        [--symmetry OPS.txt] [--pixel 3] [--max-particles 2000] [--compare-raw]

Independent of the pysmap projector: RELION's definitions (A = Euler_angles2matrix(rot, tilt, psi),
2-D transform F2(k) = F3(Aᵀ[kx, ky, 0]), x = column, y = row; image centre = coordinate − origin), a
CTF computed here from rlnDefocusU/V/Angle, voltage, Cs and amplitude contrast, and direct Fourier
insertion (trilinear, 2× oversampled) with CTF weighting. Each particle image is normalised (background
outside 0.75 of the box radius: mean 0, SD 1), multiplied by a soft circular window (0.85-0.99 of the
box radius, as in cryoSPARC's real-space windowing) and Fourier-cropped to --pixel. Two half-maps from
rlnRandomSubset (otherwise alternate particles).

--symmetry: a search with a symmetryFile reports every orientation inside one asymmetric unit, so all
viewing directions fall in 1/|G| of the sphere. The images are then slices near one plane of Fourier
space, and a C1 reconstruction leaves the cone around the common viewing direction empty (streaks
along it). With the same rotations file, every image is also inserted at its |G| equivalent poses
A Mᵀ (symmetry expansion in the template's own frame).

Reports the correlation of the map with the template (same pixel, low-passed to the half-map
resolution), the half-map FSC and the FSC against the template. --compare-raw also reconstructs the
same particles without normalisation (images as written in the stack, windowed): a large mean in the
images then becomes a disk in every image and a filled ball in the map.
Writes <out>/map.mrc, half1.mrc, half2.mrc, template.mrc (template at the map's pixel), [map_raw.mrc],
check.png (central slices and a projection of every map), fsc.png and check.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # repo root (smap)
sys.path.insert(0, str(Path(__file__).resolve().parent))         # tools/


def wavelength_A(kv: float) -> float:
    """Relativistic electron wavelength (A) at an acceleration voltage in kV."""
    v = kv * 1e3
    return 12.2643 / math.sqrt(v * (1 + 0.978466e-6 * v))


def euler_matrix(rot, tilt, psi) -> np.ndarray:
    """RELION Euler_angles2matrix (degrees)."""
    a, b, g = (math.radians(float(v)) for v in (rot, tilt, psi))
    ca, sa, cb, sb, cg, sg = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(g), math.sin(g)
    cc, cs, sc, ss = cb * ca, cb * sa, sb * ca, sb * sa
    return np.array([[cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb],
                     [-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb],
                     [sc, ss, cb]])


def ctf_2d(kx, ky, dfu, dfv, ang_deg, kv, cs_mm, amp) -> np.ndarray:
    """-(sqrt(1-A^2) sin(chi) + A cos(chi)); k in 1/A, defocus in A (underfocus positive)."""
    lam = wavelength_A(kv)
    k2 = kx ** 2 + ky ** 2
    th = np.arctan2(ky, kx)
    df = 0.5 * (dfu + dfv) + 0.5 * (dfu - dfv) * np.cos(2 * (th - math.radians(ang_deg)))
    chi = math.pi * lam * df * k2 - 0.5 * math.pi * cs_mm * 1e7 * lam ** 3 * k2 ** 2
    return -(math.sqrt(1 - amp ** 2) * np.sin(chi) + amp * np.cos(chi))


def soft_disk(n: int, r0: float, r1: float) -> np.ndarray:
    """Circular window (n, n): 1 inside r0, raised-cosine edge down to 0 at r1 (pixels from n // 2)."""
    g = np.arange(n) - n // 2
    r = np.hypot(*np.meshgrid(g, g, indexing="ij"))
    w = np.clip((r1 - r) / max(r1 - r0, 1e-6), 0, 1)
    return 0.5 - 0.5 * np.cos(math.pi * w)


def fourier_crop(img: np.ndarray, m: int) -> np.ndarray:
    """Real image (n, n) -> (m, m) by cropping its centred transform (m <= n)."""
    n = img.shape[0]
    F = np.fft.fftshift(np.fft.fft2(img))
    c, h = n // 2, m // 2
    F = F[c - h:c - h + m, c - h:c - h + m]
    return np.real(np.fft.ifft2(np.fft.ifftshift(F))) * (m / n) ** 2        # keep the pixel values' scale


def resample_volume(V: np.ndarray, pix_in: float, pix_out: float, n_out: int) -> np.ndarray:
    """Fourier crop/pad of a cubic map to pix_out, then centred crop/pad to n_out (real space)."""
    n = V.shape[0]
    m = int(round(n * pix_in / pix_out))
    m += m % 2
    F = np.fft.fftshift(np.fft.fftn(V))
    G = np.zeros((m, m, m), complex)
    a, b = n // 2, m // 2
    h = min(a, b)
    G[b - h:b + h, b - h:b + h, b - h:b + h] = F[a - h:a + h, a - h:a + h, a - h:a + h]
    W = np.real(np.fft.ifftn(np.fft.ifftshift(G))) * (m / n) ** 3
    out = np.zeros((n_out,) * 3)
    s = min(m, n_out)
    i, j = m // 2 - s // 2, n_out // 2 - s // 2
    out[j:j + s, j:j + s, j:j + s] = W[i:i + s, i:i + s, i:i + s]
    return out


def fsc(a: np.ndarray, b: np.ndarray, pix: float):
    """Fourier shell correlation of two cubic maps -> (spatial frequency in 1/A, FSC) per shell."""
    n = a.shape[0]
    A, B = np.fft.fftshift(np.fft.fftn(a)), np.fft.fftshift(np.fft.fftn(b))
    g = np.arange(n) - n // 2
    z, y, x = np.meshgrid(g, g, g, indexing="ij")
    sh = np.round(np.sqrt(x ** 2 + y ** 2 + z ** 2)).astype(int).ravel()       # shell index of every voxel
    num = np.bincount(sh, (A * np.conj(B)).real.ravel())
    da = np.bincount(sh, (np.abs(A) ** 2).ravel())
    db = np.bincount(sh, (np.abs(B) ** 2).ravel())
    k = np.arange(1, n // 2)                                                    # shells up to Nyquist
    f = num[k] / np.sqrt(da[k] * db[k] + 1e-30)
    return k / (n * pix), f


def resolution(freq, f, thr) -> float:
    """Resolution (A) of the last shell before the curve first drops below thr (Nyquist if it never does)."""
    below = np.nonzero(f < thr)[0]
    if not len(below):
        return 1 / freq[-1]
    i = below[0]
    return 1 / freq[max(i - 1, 0)]


def lowpass(V: np.ndarray, pix: float, res_A: float) -> np.ndarray:
    """Butterworth low-pass of a cubic map with its half-amplitude point at res_A."""
    n = V.shape[0]
    g = np.fft.fftfreq(n, d=pix)
    z, y, x = np.meshgrid(g, g, g, indexing="ij")
    k = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    filt = 1 / (1 + (k * res_A) ** 8)
    return np.real(np.fft.ifftn(np.fft.fftn(V) * filt))


class Inserter:
    """Direct Fourier reconstruction: CTF-weighted insertion of image transforms into a 3-D grid.

    The grid is (m, m, m) with m = 2 x image box (2x oversampling: the images are zero-padded to m),
    indexed [z, y, x] with the origin at m // 2, as an mrc map. Every image contributes
    sum(CTF x F2) to the numerator and sum(CTF^2) to the weight of the eight voxels around each of
    its slice points (trilinear). One accumulator pair per half-set."""

    def __init__(self, box: int, n_half: int = 2, ops=None):
        """box: image size after Fourier cropping; ops: symmetry operators (default: the identity)."""
        self.ops = np.asarray(ops if ops is not None else np.eye(3)[None], dtype=np.float64)
        self.b, self.m = box, 2 * box
        m = self.m
        self.num = [np.zeros(m ** 3, complex) for _ in range(n_half)]
        self.w = [np.zeros(m ** 3) for _ in range(n_half)]
        g = np.arange(m) - m // 2
        ky, kx = np.meshgrid(g, g, indexing="ij")                    # rows = y, cols = x
        keep = np.hypot(kx, ky) < m // 2 - 1                         # inside Nyquist: the +1 neighbours stay in the grid
        self.kx, self.ky = kx[keep].astype(float), ky[keep].astype(float)
        self.keep = keep
        self.buf = [[[], [], []] for _ in range(n_half)]             # pending (voxel index, value, weight) per half

    def add(self, F2: np.ndarray, ctf: np.ndarray, A: np.ndarray, half: int):
        """Insert one image; with symmetry operators M, also at the equivalent poses A Mᵀ (symmetry
        expansion: the slice of pose A Mᵀ is M Aᵀ[kx, ky, 0])."""
        m, c = self.m, self.m // 2
        base = A.T @ np.stack([self.kx, self.ky, np.zeros_like(self.kx)])         # (x, y, z) in the 3-D grid
        x, y, z = np.einsum("kij,jn->ikn", self.ops, base).reshape(3, -1) + c
        x0, y0, z0 = np.floor(x).astype(int), np.floor(y).astype(int), np.floor(z).astype(int)
        fx, fy, fz = x - x0, y - y0, z - z0
        k = len(self.ops)
        val = np.tile(F2[self.keep] * ctf[self.keep], k)
        wt = np.tile(ctf[self.keep] ** 2, k)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    t = (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * (fz if dz else 1 - fz)
                    idx = ((z0 + dz) * m + (y0 + dy)) * m + (x0 + dx)
                    self.buf[half][0].append(idx)
                    self.buf[half][1].append(val * t)
                    self.buf[half][2].append(wt * t)
        if sum(len(v) for v in self.buf[half][0]) >= 2e7:
            self.flush(half)

    def flush(self, half: int):
        """Add the pending contributions of one half-set to its accumulators."""
        b = self.buf[half]
        if not b[0]:
            return
        idx, v, w = np.concatenate(b[0]), np.concatenate(b[1]), np.concatenate(b[2])
        n = self.m ** 3
        self.num[half] += np.bincount(idx, v.real, n) + 1j * np.bincount(idx, v.imag, n)
        self.w[half] += np.bincount(idx, w, n)
        self.buf[half] = [[], [], []]

    def volume(self, halves) -> np.ndarray:
        """Map (b, b, b) from the listed half-sets ([0], [1] or [0, 1])."""
        for h in range(len(self.num)):
            self.flush(h)
        m, b = self.m, self.b
        num = sum(self.num[h] for h in halves).reshape(m, m, m)
        w = sum(self.w[h] for h in halves).reshape(m, m, m)
        # Wiener-like division: sum(CTF F) / (sum(CTF^2) + eps); eps keeps voxels with little weight
        # (CTF zeros, sparse coverage) from being amplified.
        eps = 0.02 * np.mean(w[w > 0]) if np.any(w > 0) else 1.0
        F = num / (w + eps)
        # A real map has F(-k) = conj(F(k)); only slices were inserted, so average each voxel with its
        # Friedel mate (index i <-> m - i about the origin at m // 2, i.e. reverse and roll by one).
        F = 0.5 * (F + np.conj(F[::-1, ::-1, ::-1] if m % 2 else np.roll(F[::-1, ::-1, ::-1], 1, axis=(0, 1, 2))))
        V = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(F))))
        # Trilinear interpolation in Fourier space multiplies the map by sinc^2 along each axis: undo it.
        g = (np.arange(m) - m // 2) / m
        s = np.sinc(g) ** 2
        V /= np.maximum(s[:, None, None] * s[None, :, None] * s[None, None, :], 1e-3)
        c, h = m // 2, b // 2
        return V[c - h:c - h + b, c - h:c - h + b, c - h:c - h + b]        # central box: drop the padding


def main(argv=None) -> int:
    import mrcfile
    from star_to_pars import particles

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("star")
    ap.add_argument("--template", default=None, help="template potential .mrc (e.g. the search's *_SP.mrc)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--pixel", type=float, default=3.0, help="pixel of the check map (A; default 3)")
    ap.add_argument("--max-particles", type=int, default=2000, help="random subset (default 2000; 0 = all)")
    ap.add_argument("--compare-raw", action="store_true", help="also reconstruct without normalising the images")
    ap.add_argument("--template-pixel", type=float, default=None, help="template pixel (A; default: its header)")
    ap.add_argument("--diameter-A", type=float, default=None, help="particle diameter for the correlation mask")
    ap.add_argument("--symmetry", default=None,
                    help="rotations file (the search's symmetryFile): insert every image at its |G| equivalent poses")
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    # ---- particles and optics (one optics group; values of the first row)
    star = Path(a.star).expanduser()
    rows = particles(star)
    if a.max_particles and len(rows) > a.max_particles:
        sel = np.sort(np.random.default_rng(0).choice(len(rows), a.max_particles, replace=False))
        rows = [rows[i] for i in sel]
    r0 = rows[0]
    pix = float(r0.get("rlnImagePixelSize") or r0.get("rlnMicrographOriginalPixelSize"))
    kv, cs, amp = float(r0["rlnVoltage"]), float(r0["rlnSphericalAberration"]), float(r0["rlnAmplitudeContrast"])
    stacks = {}

    def image(name: str) -> np.ndarray:
        """Image of 'j@stack' (j is 1-based; stack paths are relative to the STAR file)."""
        j, path = name.split("@", 1)
        p = Path(path)
        p = p if p.is_absolute() else star.parent / p
        if p not in stacks:
            stacks[p] = mrcfile.mmap(p, permissive=True, mode="r")
        d = stacks[p].data
        return np.asarray(d[int(j) - 1] if d.ndim == 3 else d, dtype=np.float64)

    # ---- geometry: images of n_in px are Fourier-cropped to b px (even), i.e. to p_out A/px
    n_in = image(r0["rlnImageName"]).shape[0]
    b = int(round(n_in * pix / a.pixel))
    b += b % 2
    b = min(b, n_in - n_in % 2)
    p_out = n_in * pix / b
    win = soft_disk(n_in, 0.85 * n_in / 2, 0.99 * n_in / 2)
    g = np.arange(n_in) - n_in // 2
    rr = np.hypot(*np.meshgrid(g, g, indexing="ij"))
    bg = rr > 0.75 * n_in / 2                                        # background ring for the normalisation
    from smap import io as sio
    ops = sio.read_rotations_file(a.symmetry) if a.symmetry else None
    ins = Inserter(b, ops=ops)
    ins_raw = Inserter(b, ops=ops) if a.compare_raw else None
    m = ins.m
    f = np.fft.fftshift(np.fft.fftfreq(m, d=p_out))
    KY, KX = np.meshgrid(f, f, indexing="ij")                        # 1/A, for the CTF on the padded grid
    fi = np.fft.fftshift(np.fft.fftfreq(n_in))
    FY, FX = np.meshgrid(fi, fi, indexing="ij")                      # cycles/px, for origin shifts
    stats = []

    # ---- insert every image: normalise, window, shift to its origin, crop, pad, transform
    for i, r in enumerate(rows):
        img = image(r["rlnImageName"])
        stats.append((float(img.mean()), float(img[bg].std())))
        ox = float(r.get("rlnOriginXAngst", 0) or 0) / pix
        oy = float(r.get("rlnOriginYAngst", 0) or 0) / pix
        A = euler_matrix(r["rlnAngleRot"], r["rlnAngleTilt"], r["rlnAnglePsi"])
        ctf = ctf_2d(KX, KY, float(r["rlnDefocusU"]), float(r["rlnDefocusV"]), float(r.get("rlnDefocusAngle", 0)),
                     kv, cs, amp)
        half = int(r["rlnRandomSubset"]) - 1 if "rlnRandomSubset" in r else i % 2
        versions = [(ins, (img - img[bg].mean()) / max(img[bg].std(), 1e-12))]
        if ins_raw is not None:
            versions.append((ins_raw, img))                           # the same image as stored, for --compare-raw
        for target, im in versions:
            im = im * win
            if ox or oy:                                              # centre = coordinate - origin
                im = np.real(np.fft.ifft2(np.fft.fft2(im) * np.exp(-2j * math.pi * np.fft.ifftshift(FX * ox + FY * oy))))
            small = fourier_crop(im, b)
            pad = np.zeros((m, m))
            pad[m // 2 - b // 2:m // 2 + b // 2, m // 2 - b // 2:m // 2 + b // 2] = small
            F2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pad)))
            target.add(F2, ctf, A, half)
        if (i + 1) % 250 == 0:
            print(f"{i + 1}/{len(rows)} particles", flush=True)

    # ---- maps, half-map FSC and report
    h1, h2, full = ins.volume([0]), ins.volume([1]), ins.volume([0, 1])
    fr, f_half = fsc(h1, h2, p_out)
    res = resolution(fr, f_half, 0.143)
    rep = dict(particles=len(rows), symmetry_operators=len(ins.ops), pixel_A=round(p_out, 4), box=b,
               half_map_fsc0143_A=round(res, 2),
               image_mean_median=round(float(np.median([s[0] for s in stats])), 4),
               image_background_sd_median=round(float(np.median([s[1] for s in stats])), 4))
    for name, V in (("map", full), ("half1", h1), ("half2", h2)):
        with mrcfile.new(out / f"{name}.mrc", overwrite=True) as mm:
            mm.set_data(V.astype(np.float32))
            mm.voxel_size = p_out
    vols = [("map (normalised images)", full)]
    raw = None
    if ins_raw is not None:
        raw = ins_raw.volume([0, 1])
        with mrcfile.new(out / "map_raw.mrc", overwrite=True) as mm:
            mm.set_data(raw.astype(np.float32))
            mm.voxel_size = p_out
        vols.append(("map (images as in the stack)", raw))
    # ---- comparison with the template: same pixel and box, both low-passed to the half-map resolution
    T = None
    if a.template:
        with mrcfile.open(a.template, permissive=True) as mt:
            T0, tp = np.asarray(mt.data, dtype=np.float64), float(a.template_pixel or mt.voxel_size.x)
        if tp <= 0:
            raise SystemExit(f"{a.template}: no pixel size in the header; give --template-pixel")
        rep["template_pixel_A"] = tp
        T = resample_volume(T0 - np.median(T0), tp, p_out, b)
        with mrcfile.new(out / "template.mrc", overwrite=True) as mm:
            mm.set_data(T.astype(np.float32))
            mm.voxel_size = p_out
        rad = (a.diameter_A / 2 if a.diameter_A else 0.45 * b * p_out) / p_out
        gg = np.arange(b) - b // 2
        z, y, x = np.meshgrid(gg, gg, gg, indexing="ij")
        mask = np.sqrt(x ** 2 + y ** 2 + z ** 2) < rad * 1.1                    # sphere just outside the particle
        lp = max(res, 2 * p_out)                                                # not finer than Nyquist
        Tl = lowpass(T, p_out, lp)
        ft, f_t = fsc(full, T, p_out)
        rep["fsc_vs_template_0.5_A"] = round(resolution(ft, f_t, 0.5), 2)
        for (_, V), key in zip(vols, ("correlation_with_template", "correlation_with_template_raw")):
            Vl = lowpass(V, p_out, lp)
            rep[key] = round(float(np.corrcoef(Vl[mask], Tl[mask])[0, 1]), 3)
        if rep["correlation_with_template"] < 0:
            print("NOTE: negative correlation: the map has the opposite contrast to the template")
    (out / "check.json").write_text(json.dumps(rep, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = vols + ([("template", T)] if T is not None else [])
    fig, ax = plt.subplots(len(panels), 4, figsize=(13, 3.2 * len(panels)), squeeze=False)
    c = b // 2
    for i, (label, V) in enumerate(panels):
        Vd = lowpass(V, p_out, max(res, 2 * p_out))
        for j, (sl, nm) in enumerate(((Vd[c], "z = 0"), (Vd[:, c], "y = 0"), (Vd[:, :, c], "x = 0"))):
            ax[i, j].imshow(sl, cmap="gray")
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_title(f"{label}: {nm}" if j == 0 else nm, fontsize=8)
        proj = Vd.sum(axis=0)
        ax[i, 3].imshow(proj, cmap="gray")
        ax[i, 3].set_title("projection along z", fontsize=8)
        ax[i, 3].set_xticks([])
        ax[i, 3].set_yticks([])
    fig.suptitle(f"{len(rows)} particles" + (f" x {len(ins.ops)} symmetry operators" if len(ins.ops) > 1 else "")
                 + f", {p_out:.2f} Å/px, half-map FSC 0.143: {res:.1f} Å"
                 + (f"; correlation with template {rep.get('correlation_with_template', float('nan')):.2f}" if T is not None else ""),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "check.png", dpi=110, facecolor="white")
    plt.close(fig)
    fig, axf = plt.subplots(figsize=(5, 3.2))
    axf.plot(fr, f_half, label="half-maps", color="#2a78d6")
    if T is not None:
        axf.plot(ft, f_t, label="map vs template", color="#eb6834")
    axf.axhline(0.143, color="#999999", lw=0.6)
    axf.axhline(0.5, color="#cccccc", lw=0.6)
    axf.set_xlabel("spatial frequency (1/Å)")
    axf.set_ylabel("FSC")
    axf.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fsc.png", dpi=110, facecolor="white")
    plt.close(fig)
    for k, v in rep.items():
        print(f"{k}: {v}")
    print(f"-> {out}/check.png, fsc.png, map.mrc")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
