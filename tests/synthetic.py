"""Shared synthetic data for the end-to-end tests.

A random 'atomic' volume (400 Gaussian blobs in a sphere) is projected at a known grid
orientation with the port's own template generator, CTF-filtered, scaled to a chosen
matched-filter SNR and placed at a known pixel of a white-noise image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from smap import io as sio
from smap.ctf import ctf
from smap.search_global import make_templates, prepare

N = 48     # volume / template edge (pixels)
IM = 144   # image edge (pixels; 2^4*3^2, so no FFT padding)


def volume(rng) -> np.ndarray:
    """(N, N, N) float32 volume of 400 Gaussian 'atoms' inside a sphere of radius 15."""
    z, y, x = np.mgrid[:N, :N, :N].astype(np.float64)
    pts = rng.uniform(-1, 1, (4000, 3))
    pts = pts[np.linalg.norm(pts, axis=1) < 1][:400] * 15 + 24
    vol = np.zeros((N, N, N), np.float32)
    for pz, py, px in pts:
        vol += np.exp(-((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) / 1.5).astype(np.float32)
    return vol


def write_par(d: Path, img, n_cores: int = 2, extra: str = "") -> Path:
    """search.par for the synthetic data (20 deg grid, one defocus plane)."""
    p = d / "s.par"
    p.write_text(
        "function search_global\n"
        f"nCores {n_cores}\n"
        f"imageFile {img}\n"
        f"structureFile {d / 'vol.mrc'}\n"
        f"outputDir {d / 'out'}\n"
        "aPerPix 2.0\n"
        "defocus 15000.0 14000.0 30.0\n"
        "angle_inc 20\n"
        "T_sample 0\n"
        "arbThr 4.0\n"
        "margin_pix 0\n" + extra
    )
    return p


def make_tile(d: Path, k: int, r0: int, c0: int, snr: float, phase_scale: float = 1.0):
    """Write vol.mrc and img.mrc in ``d`` with the rotation-``k`` projection of the volume
    centred at 0-based pixel (r0, c0). ``phase_scale`` scales the potential (use ~0.002 to
    keep the exit-wave template of the optimisation stage in the weak-phase regime)."""
    rng = np.random.default_rng(0)
    sio.mw(np.transpose(volume(rng) * phase_scale, (1, 2, 0)), d / "vol.mrc", 2.0)
    sio.mw(rng.standard_normal((IM, IM)).astype(np.float32), d / "dummy.mrc", 2.0)
    st0 = prepare(write_par(d, d / "dummy.mrc"), write_files=False)
    img = np.zeros((IM, IM))
    h = N // 2
    img[r0 - h:r0 - h + N, c0 - h:c0 - h + N] = make_templates(st0, [k])[0]
    Ci = np.imag(np.fft.ifftshift(ctf(st0.defocus, [IM, IM], st0.params)[:, :, 0]))
    sig = np.real(np.fft.ifft2(np.fft.fft2(img) * Ci))
    sig *= snr / np.linalg.norm(sig)
    sio.mw((sig + rng.standard_normal((IM, IM))).astype(np.float32), d / "img.mrc", 2.0)
    return d / "img.mrc"
