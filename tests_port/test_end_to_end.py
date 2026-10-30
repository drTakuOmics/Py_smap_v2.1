"""End-to-end functional test: prepare -> run_job (2 jobs) -> combine on synthetic data.

A random atom-like volume is projected at a known grid orientation, CTF-filtered and
inserted at a known pixel position in white noise. The search must return that
rotation index and position as the top entry of highVals.txt.
"""

import numpy as np
import pytest

from smap import io as sio
from smap.ctf import ctf
from smap.search_global import combine, make_templates, prepare, run_job

N = 48
IM = 144


def _volume(rng):
    z, y, x = np.mgrid[:N, :N, :N].astype(np.float64)
    pts = rng.uniform(-1, 1, (4000, 3))
    pts = pts[np.linalg.norm(pts, axis=1) < 1][:400] * 15 + 24
    vol = np.zeros((N, N, N), np.float32)
    for pz, py, px in pts:
        vol += np.exp(-((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) / 1.5).astype(np.float32)
    return vol


def _write_par(d, img):
    p = d / "s.par"
    p.write_text(
        "function search_global\n"
        "nCores 2\n"
        f"imageFile {img}\n"
        f"structureFile {d / 'vol.mrc'}\n"
        f"outputDir {d / 'out'}\n"
        "aPerPix 2.0\n"
        "defocus 15000.0 14000.0 30.0\n"
        "angle_inc 20\n"
        "T_sample 0\n"
        "arbThr 4.0\n"
        "margin_pix 0\n"
    )
    return p


@pytest.mark.parametrize("k,r0,c0", [(137, 72, 72), (1500, 50, 95)])
def test_recovers_inserted_particle(tmp_path, k, r0, c0):
    rng = np.random.default_rng(0)
    sio.mw(np.transpose(_volume(rng), (1, 2, 0)), tmp_path / "vol.mrc", 2.0)
    sio.mw(rng.standard_normal((IM, IM)).astype(np.float32), tmp_path / "dummy.mrc", 2.0)
    st0 = prepare(_write_par(tmp_path, tmp_path / "dummy.mrc"), write_files=False)

    tpl = make_templates(st0, [k])[0]
    img = np.zeros((IM, IM))
    h = N // 2
    img[r0 - h:r0 - h + N, c0 - h:c0 - h + N] = tpl
    Ci = np.imag(np.fft.ifftshift(ctf(st0.defocus, [IM, IM], st0.params)[:, :, 0]))
    sig = np.real(np.fft.ifft2(np.fft.fft2(img) * Ci))
    sig *= 12.0 / np.linalg.norm(sig)
    sio.mw((sig + rng.standard_normal((IM, IM))).astype(np.float32), tmp_path / "img.mrc", 2.0)

    st = prepare(_write_par(tmp_path, tmp_path / "img.mrc"))
    for j in (1, 2):
        run_job(st, j, device="cpu", batch_size=8)
    combine(st)
    hv = np.loadtxt(tmp_path / "out" / "highVals.txt", ndmin=2)
    top = hv[np.argmax(hv[:, 1])]
    assert int(top[0]) == k
    assert (int(top[2]), int(top[3])) == (r0 + 1, c0 + 1)  # 1-based, MATLAB convention
